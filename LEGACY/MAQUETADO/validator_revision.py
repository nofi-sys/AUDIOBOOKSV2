"""
validator_revision.py — Validador compacto para respuestas de corrección (GPT-5)

Uso (ejemplos):
  python validator_revision.py response.json original.txt config.json
  # o pasando strings directamente:
  python validator_revision.py --strings '{"estado":"ok","revision":"…","observaciones":""}' "Texto original…" config.json
"""

import json, re, sys

ALLOWED_ESTADOS = {"ok","dudoso","mal"}

def load_arg(i):
    if i < len(sys.argv): 
        return sys.argv[i]
    raise SystemExit("Faltan argumentos. Ver uso en el docstring.")

def read_file(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def parse_inputs():
    if len(sys.argv) >= 5 and sys.argv[1] == "--strings":
        resp = json.loads(sys.argv[2])
        original = sys.argv[3]
        config = json.loads(read_file(sys.argv[4]))
    elif len(sys.argv) >= 4:
        resp = json.loads(read_file(sys.argv[1]))
        original = read_file(sys.argv[2])
        config = json.loads(read_file(sys.argv[3]))
    else:
        raise SystemExit("Uso: validator_revision.py response.json original.txt config.json  OR  --strings <json> <original> config.json")
    return resp, original, config

def only_allowed_quotes(text, style):
    # Devuelve error si encuentra comillas no permitidas
    if style == "angulares":
        if re.search(r"[“”]", text): return "Se hallaron comillas inglesas en style=angulares."
    elif style == "inglesas":
        if re.search(r"[«»]", text): return "Se hallaron comillas angulares en style=inglesas."
    # 'mantener': no se valida tipo, solo coherencia básica (no mezclar)
    return None

def mixed_quotes(text):
    return bool(re.search(r"[«»]", text) and re.search(r"[“”]", text))

def has_html(text):
    return bool(re.search(r"<[^>]+>", text))

def dialogue_policy_errors(text, style):
    errs = []
    if style == "forzar_raya":
        # No debe iniciar diálogos con comillas inglesas/angulares
        if re.search(r'(^|\n)\s*[«“"]', text):
            errs.append("DIALOGUE_STYLE=forzar_raya: se detectaron diálogos iniciados con comillas.")
        # Debe haber rayas si hay intervenciones (heurística: signos ¿? seguidos de habla)
        # y no mezclar comillas de apertura con parlamentos largos
        # Chequeo de espacios tras raya de apertura (— )
        if re.search(r"(^|\n)—\s", text):
            errs.append("Espacio indebido tras raya de apertura de diálogo (— ).")
    # Chequeos genéricos para rayas en incisos: evitar '— texto' dentro del inciso
    if re.search(r"—\s+[,.?!;:]", text):
        errs.append("Espacio indebido tras raya antes de signo de puntuación.")
    return errs

def validate(resp, original, config):
    errors = []
    # Schema básico
    for k in ("estado","revision","observaciones"):
        if k not in resp: errors.append(f"Falta clave '{k}'.")
    if errors: return errors
    if resp["estado"] not in ALLOWED_ESTADOS:
        errors.append("Valor de 'estado' inválido.")
    if resp["estado"] == "ok" and resp["revision"] != original:
        errors.append("estado=ok pero 'revision' difiere del original.")
    # Sin HTML
    if has_html(resp["revision"]):
        errors.append("Se detectó HTML/etiquetas en 'revision'.")
    # Política de comillas
    quote_style = config.get("defaults",{}).get("QUOTE_STYLE","mantener")
    qerr = only_allowed_quotes(resp["revision"], quote_style)
    if qerr: errors.append(qerr)
    if quote_style != "mantener" and mixed_quotes(resp["revision"]):
        errors.append("Se mezclaron comillas inglesas y angulares.")
    # Política de diálogo
    dialogue_style = config.get("defaults",{}).get("DIALOGUE_STYLE","mantener")
    errors.extend(dialogue_policy_errors(resp["revision"], dialogue_style))
    # Observaciones vacías cuando ok
    if resp["estado"] == "ok" and (resp.get("observaciones") or "") != "":
        errors.append("estado=ok requiere 'observaciones' vacías.")
    return errors

def main():
    resp, original, config = parse_inputs()
    errs = validate(resp, original, config)
    if errs:
        print("INVALIDO")
        for e in errs: print(f"- {e}")
        sys.exit(1)
    print("OK")

if __name__ == "__main__":
    main()
