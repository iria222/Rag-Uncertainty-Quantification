from citations import format_context_label
from config import GEN_MODEL_NAME, QUANTIZATION, APPLY_SELF_EVALUATION
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from utils import free_model, debug_print
from self_evaluation import evaluate_answer_by_prompt
import torch
import time
import re


SYSTEM_PROMPT = (
    "Eres un asistente de investigación académico riguroso."
    "Tu única fuente de verdad es el fragmento proporcionado en el apartado 'CONTEXTO'.\n"
    "REGLAS:\n"
    "1. Responde de forma directa, concisa y basada exclusivamente en el contexto.\n"
    "2. No inventes información ni utilices conocimientos externos al contexto.\n"
    "3. Cita las afirmaciones utilizando exclusivamente números entre corchetes que se correspondan con el índice del fragmento utilizado (ej: [1], [2]).\n"
    "4. Al final de tu respuesta, debes incluir obligatoriamente una sección titulada 'Referencias:' donde mapees cada número utilizado con su fuente exacta en el formato:"
    "'[1] Archivo: nombre_archivo.pdf, Pág. X'.\n"
    "La única excepción a esta regla será cuando te abstengas de responder, en cuyo caso no deberás incluir la sección 'Referencias'./"
    "5. Si la respuesta no se encuentra o no se puede deducir con certeza del contexto, debes abstenerte respondiendo exactamente con el siguiente texto,"
    "sin añadir ninguna sección de 'Referencias:' ni texto adicional: "
    "'[ABSTENCION] No dispongo de suficiente información en el corpus para responder."
)

USER_PROMPT_TEMPLATE = """CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

_global_model = None
_global_processor = None

_ABSTENTION_MESSAGE = "[ABSTENCION] No dispongo de suficiente información en el corpus para responder.\n"


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
    has_cuda = torch.cuda.is_available()
    debug_print(f"CUDA available: {has_cuda}")

    quant_config = None
    model_kwargs = {}

    if has_cuda:
        debug_print(f"Loading model on GPU with quantization: {quantization}")
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs["device_map"] = {"": 0}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        
        debug_print("No GPU detected. Falling back to CPU (Full precision). THIS IS NOT RECOMMENDED")
        model_kwargs["device_map"] = "cpu"
        model_kwargs["torch_dtype"] = torch.float32 

    processor = AutoProcessor.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForMultimodalLM.from_pretrained(
        GEN_MODEL_NAME, **model_kwargs
    )

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
    
def generate_answer(query: str, reranked_chunks: list[dict]) -> tuple[str,float,float, float]:
    """Generates an answer for a specific question

    Args:
        query (str): input question to answer
        reranked_chunks (list[dict]): retrieved chunks

    Returns:
        tuple[str,float,float,str]: processed question, model's load time in seconds, generation time in seconds, logit probabilities
    """
    
    load_time = 0.0
    generation_time = 0.0
    prob_true = None
    
    if not reranked_chunks:
        return (_ABSTENTION_MESSAGE, load_time, generation_time, prob_true)  
    context = build_context_for_prompt(reranked_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    load_time, model, processor = get_model_and_processor()
    
    
    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text= f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        
    inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)
    
    debug_print("Starting text generation...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 512,
            temperature = 0.2,
            do_sample = True
        )
        
    end_time = time.perf_counter()
    generation_time = end_time - start_time
    debug_print(f"Generation completed in {generation_time:.2f} seconds.")
        
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    raw_answer = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Self evaluation by prompting
    if APPLY_SELF_EVALUATION and "[ABSTENCION]" not in raw_answer:
        context_str = build_context_for_prompt(reranked_chunks)
        prob_true, must_abstain = evaluate_answer_by_prompt(query, context_str, raw_answer, model, processor)
        debug_print(f"[Experimento Self-Evaluation] P(True) estimado: {prob_true: .4f}")
        
        if must_abstain:
            raw_answer = _ABSTENTION_MESSAGE
            
    answer = postprocess_answer(raw_answer)
    
    return answer, load_time, generation_time, prob_true


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

def postprocess_answer(raw_answer:str) -> str:
    """Cleans answer

    Args:
        raw_answer (str): answer generated by the llm

    Returns:
        str: cleaned answer
    """
    
    if "[ABSTENCION]" in raw_answer:
        cleaned_answer = _ABSTENTION_MESSAGE
        return cleaned_answer
    
    abstention_patterns = [
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
    
    for pattern in abstention_patterns:
        if re.search(pattern,raw_answer.lower()):
            return _ABSTENTION_MESSAGE
        
    return raw_answer
