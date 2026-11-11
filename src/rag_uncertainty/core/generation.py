from rag_uncertainty.core.citations import format_context_label
from rag_uncertainty.config import GEN_MODEL_NAME, QUANTIZATION, APPLY_SELF_EVALUATION, SELF_EVAL_THRESHOLD
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from rag_uncertainty.utils.utils import free_model, debug_print, validate_quantization
from rag_uncertainty.evaluation.self_evaluation import compute_p_true
from rag_uncertainty.evaluation.uncertainty.metrics import compute_mean_token_entropy
import torch
import time
import re


SYSTEM_PROMPT = (
    "Eres un asistente de investigación académico riguroso.\n"
    "Tu única fuente de verdad es el fragmento proporcionado en el apartado 'CONTEXTO'.\n"
    "REGLAS:\n"
    "1. Responde de forma directa, concisa y basada exclusivamente en el contexto.\n"
    "2. No inventes información ni utilices conocimientos externos al contexto.\n"
    "3. Cita las afirmaciones utilizando exclusivamente números entre corchetes que se correspondan con el índice del fragmento utilizado (ej: [1], [2]).\n"
    "4. Si la respuesta no se encuentra o no se puede deducir con certeza del contexto, debes abstenerte respondiendo exactamente con el siguiente texto, "
    "'[ABSTENCION] No dispongo de suficiente información en el corpus para responder.'\n"
    "5. No repitas la pregunta ni el contexto, ni uses encabezados como 'PREGUNTA:' o 'RESPUESTA:'. Empieza directamente con la respuesta.\n"
)

USER_PROMPT_TEMPLATE = """CONTEXTO:
{context}

PREGUNTA:
{query}"""

_global_model = None
_global_processor = None

_ABSTENTION_MESSAGE = "[ABSTENCION] No dispongo de suficiente información en el corpus para responder.\n"

# Abstention patterns are only searched at the start of the answer to avoid turn correct answers into abstentions
_ABSTENTION_PATTERNS = [
    r"el contexto proporcionado no contiene",
    r"no se proporciona informaci[oó]n",
    r"no se encuentra informaci[oó]n",
    r"no se menciona en el contexto",
    r"no se menciona en el corpus",
    r"no hay informaci[oó]n disponible en el contexto",
    r"no se encuentra en el contexto",
    r"el contexto no aporta",
    r"no dispongo de suficiente informaci[oó]n"
]

_ABSTENTION_HEAD_CHARS = 250

def build_context_for_prompt(reranked_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(reranked_chunks, 1):
        label = format_context_label(chunk, index=i)
        text = chunk.get("text", "").strip()
        context_blocks.append(f"{label}\n{text}")
        
    return "\n\n".join(context_blocks)


def load_generation_model(quantization: str = QUANTIZATION):
    """Load the multimodal generation model with an adaptive quantization strategy.

    Args:
        quantization (str): Quantization mode ("4bit", "8bit", or None for full
            precision).
    """
    validate_quantization(quantization)
         
    has_cuda = torch.cuda.is_available()
    debug_print(f"CUDA available: {has_cuda}")

    model_kwargs = {}

    if has_cuda:
        debug_print(f"Loading model on GPU with quantization: {quantization}")
        model_kwargs["device_map"] = {"":0}
        model_kwargs["dtype"] = torch.float16
        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant= True,
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
    else:
        if quantization is not None:
            debug_print(f"WARNING: quantization='{quantization}' requires a GPU and will be ignored.")
        
        debug_print("No GPU detected. Falling back to CPU (Full precision). THIS IS NOT RECOMMENDED")
        model_kwargs["device_map"] = "cpu"
        model_kwargs["dtype"] = torch.float32 

    processor = AutoProcessor.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForMultimodalLM.from_pretrained(GEN_MODEL_NAME, **model_kwargs)

    return model, processor

def free_generation_model():
    """Frees generator and processor from memory
    """
    global _global_processor, _global_model
    
    if _global_model is not None:
        debug_print("Freeing generation model from memory...")
        free_model(_global_model)
        _global_model = None
    
    if _global_processor is not None:
        del _global_processor
        _global_processor = None
        
def is_abstention(text:str) -> bool:
    """True if the answer is (explicitly or implicitly) an abstention

    Args:
        text (str): generated answer

    Returns:
        bool: whether the answer is an abstention or not
    """
    
    if "[ABSTENCION]" in text:
        return True
    head = text[:_ABSTENTION_HEAD_CHARS].lower()
    return any(re.search(p,head) for p in _ABSTENTION_PATTERNS)

def has_answer_body(text:str) -> bool:
    # Remove references section
    body = re.split(
        r"\bReferencias\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # Remove reference entries
    body = re.sub(
        r"(?m)^\s*\[\d+\]\s*Archivo:.*$",
        "",
        body
    )

    # Remove inline citation markers
    body = re.sub(r"\[\d+\]", "", body)

    return bool(re.search(r"[^\W\d_]{2,}", body))

def get_references(answer: str, retrieved_chunks: list[dict]) -> str | None:
    """Creates a references section for the answer based on the retrieved chunks.
    The citation numbers must correspond to the order in which retrieved_chunks were provided to the generator.

    Args:
        answer (str): generated answer
        retrieved_chunks (list[dict]): retrieved chunks

    Returns:
        str: references section, or None if no references were found
    """
    # Find citation identifiers ([1], [2]...)
    citation_ids = []
    
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        citation_ids += re.findall(r"\d+", group)
    citation_ids = list(dict.fromkeys(citation_ids))
    
    references = []
    
    for citation_id in citation_ids:
        index = int(citation_id) - 1
        
        if not 0 <= index < len(retrieved_chunks):
            continue
        
        chunk = retrieved_chunks[index]
        
        reference = (
            f"[{citation_id}] "
            f"Archivo: {chunk['source']}, "
            f"Pág. {chunk['page']}" 
        )
        
        references.append(reference)
    if not references:
        return None
    references_section = "\n\nReferencias:\n" + "\n".join(references)
    return references_section
    
def generate_answer(query: str, reranked_chunks: list[dict]) -> dict:
    """Generates an answer for a specific question

    Args:
        query (str): input question to answer
        reranked_chunks (list[dict]): retrieved chunks

    Returns:
        dict: dictionary with the following keys:
            - answer: the final answer (abstention or not)
            - raw_answer: the raw answer generated by the model (abstention or not)
            - references: the references section (if any)
            - load_time: time taken to load the model and processor
            - generation_time: time taken to generate the answer
            - eval_time: time taken to evaluate the answer (if self-evaluation is applied)
            - p_true: probability of the answer being true (if self-evaluation is applied)
            - mass_ab: mass assigned to abstention (if self-evaluation is applied)
            - model_abstained: whether the model abstained or not
            - self_eval_abstained: whether the self-evaluation abstained or not
            - uncertainty_metrics: dictionary with uncertainty metrics (if computed)
    """
    
    result = {
        "answer": _ABSTENTION_MESSAGE,
        "raw_answer": None,
        "references": None,
        "load_time": 0.0,
        "generation_time": 0.0,
        "eval_time": 0.0,
        "p_true": None,
        "mass_ab": None,
        "model_abstained": False,
        "empty_answer": False,
        "self_eval_abstained": False,
        "uncertainty_metrics": None
    }
    
    if not reranked_chunks:
        result["model_abstained"] = True
        return result
    
    context = build_context_for_prompt(reranked_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    load_time, model, processor = get_model_and_processor()
    result["load_time"] = load_time
    
    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text= f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        
    inputs = processor(text=prompt_text, add_special_tokens = False, return_tensors="pt").to(model.device)
    
    debug_print("Starting text generation...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 512,
            do_sample = False,
            return_dict_in_generate = True,
            output_scores = True,
            output_logits = True
        )
        
    
    end_time = time.perf_counter()
    generation_time = end_time - start_time
    result["generation_time"] = generation_time
    debug_print(f"Generation completed in {generation_time:.2f} seconds.")
    
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs.sequences[0][input_len:]
    
    # Original model output (without references section)
    raw_answer = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    result["raw_answer"] = raw_answer
    
    uncertainty_metrics = compute_uncertainty_metrics(outputs)
    result["uncertainty_metrics"] = uncertainty_metrics
    
    del outputs
    
    explicit_abstention = is_abstention(raw_answer)
    
    empty_answer = (not explicit_abstention) and (not has_answer_body(raw_answer))
    
    model_abstained = explicit_abstention or empty_answer
    result["model_abstained"] = model_abstained
    result["empty_answer"] = empty_answer
    
    if model_abstained:
        result["answer"] = _ABSTENTION_MESSAGE
        return result
    
    result["answer"] = raw_answer
    result["references"] = get_references(raw_answer, reranked_chunks)
        
    # Self evaluation by prompting
    if APPLY_SELF_EVALUATION:
        eval_start = time.perf_counter()
        ev = compute_p_true(query, context, raw_answer, model, processor, build_context_for_prompt)
        result["eval_time"] = time.perf_counter() - eval_start
        result["p_true"] = ev["p_true"]
        result["mass_ab"] = ev["mass_ab"]
        
        debug_print(f"[Self-Evaluation] P(True) = {ev["p_true"]:.4f} (masa en A/B: {ev["mass_ab"]:.3f})")
        
        if SELF_EVAL_THRESHOLD is not None and ev["p_true"] < SELF_EVAL_THRESHOLD:
            result["answer"] = _ABSTENTION_MESSAGE
            result["references"] = None
            result["self_eval_abstained"] = True
            
            
    return result


def get_model_and_processor():
    """Gets or loads generator model and processor

    Returns:
        tuple: loading time in seconds, model, processor
    """
    global _global_model
    global _global_processor
    
    load_time = 0.0
    
    if _global_processor is None or _global_model is None:
        load_start = time.perf_counter()
        model, processor = load_generation_model()
        load_end = time.perf_counter()
    
        load_time = load_end - load_start
        debug_print(f"Model loaded in {load_time:.2f} seconds.\n")
        
        
        _global_model, _global_processor = model, processor
        
    return load_time,_global_model,_global_processor


def compute_uncertainty_metrics(outputs):
    mean_token_entropy = compute_mean_token_entropy(outputs.logits)
    
    metrics = {
        "mean_token_entropy": mean_token_entropy,
    }
    
    return metrics