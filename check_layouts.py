# check_layouts_dashboards.py
import os
import importlib.util
import sys

ROOT = os.getcwd()
DASHBOARDS_ROOT = os.path.join(ROOT, "dashboards")

checks = [
    ("facebook", "facebook_layout.py", "render_facebook_dashboard"),
    ("x_predictivo", "x_layout.py", "render_x_dashboard"),
    ("benchmark", "benchmark_layout.py", "render_benchmark_dashboard"),
    ("telegram", "telegram_layout.py", "render_telegram_dashboard"),
]

def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("temp_module", path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        return ("LOAD_ERROR", str(e))

print(f"Directorio de trabajo: {ROOT}\n")
print(f"Buscando módulos dentro de: {DASHBOARDS_ROOT}\n")

all_ok = True
for pkg, fname, func in checks:
    pkg_path = os.path.join(DASHBOARDS_ROOT, pkg)
    file_path = os.path.join(pkg_path, fname)
    print(f"--- comprobando dashboards/{pkg}/{fname} (esperando función: {func}) ---")

    if not os.path.isdir(pkg_path):
        print(f"  ❌ Carpeta no encontrada: {pkg_path}")
        all_ok = False
        continue
    else:
        print(f"  ✅ Carpeta encontrada: {pkg_path}")

    init_path = os.path.join(pkg_path, "__init__.py")
    if not os.path.exists(init_path):
        print(f"  ⚠️ Falta __init__.py en dashboards/{pkg}  (recomendado: crear archivo vacío __init__.py)")
    else:
        print(f"  ✅ __init__.py encontrado")

    if not os.path.exists(file_path):
        candidates = [f for f in os.listdir(pkg_path) if f.lower() == fname.lower()]
        if candidates:
            print(f"  ❌ Archivo exacto no encontrado: {file_path}")
            print(f"     ⚠️ Existe similar (case-diff): {candidates[0]}  -> renómbralo a {fname}")
        else:
            print(f"  ❌ Archivo no encontrado: {file_path}")
        all_ok = False
        continue
    else:
        print(f"  ✅ Archivo encontrado: {file_path}")

    mod = load_module_from_path(file_path)
    if mod is None:
        print(f"  ❌ No se pudo crear spec para el archivo. Revisa permisos o sintaxis.")
        all_ok = False
        continue
    if isinstance(mod, tuple) and mod[0] == "LOAD_ERROR":
        print(f"  ❌ Error al cargar el módulo: {mod[1]}")
        all_ok = False
        continue

    if hasattr(mod, func) and callable(getattr(mod, func)):
        print(f"  ✅ Función encontrada: {func}()")
    else:
        exported = [name for name in dir(mod) if callable(getattr(mod, name))]
        print(f"  ❌ Función {func}() NO encontrada en el módulo.")
        print(f"     Funciones detectadas (filtrado): {exported[:20]}")
        all_ok = False

print("\nResumen:")
if all_ok:
    print("✅ Todo parece estar en su lugar: carpetas, archivos y funciones detectadas.")
else:
    print("❗ Hay problemas. Revisa las líneas anteriores para saber qué corregir.")
