"""
Скрипт для диагностики: решает экспортированную модель отдельно от основного кода.
Использование:
    python solve_exported_model.py log/debug_model_before_solve.pb
"""
import sys
import faulthandler

# Включаем faulthandler для получения stacktrace при crash
faulthandler.enable()

from ortools.sat.python import cp_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python solve_exported_model.py <model.pb>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    print(f"Loading model from {model_path}...", flush=True)
    
    model = cp_model.CpModel()
    
    # Читаем модель из protobuf файла
    # Способ зависит от версии OR-Tools
    try:
        # Новые версии OR-Tools
        from google.protobuf import text_format
        from ortools.sat import cp_model_pb2
        
        proto = cp_model_pb2.CpModelProto()
        with open(model_path, "rb") as f:
            proto.ParseFromString(f.read())
        model = cp_model.CpModel()
        model.Proto().CopyFrom(proto)
    except Exception as e1:
        print(f"Method 1 failed: {e1}", flush=True)
        # Альтернативный способ
        try:
            model.ImportFromFile(model_path)
        except Exception as e2:
            print(f"Method 2 failed: {e2}", flush=True)
            print("Cannot load model", flush=True)
            sys.exit(1)
    
    print(f"Model loaded. Variables: {len(model.Proto().variables)}, Constraints: {len(model.Proto().constraints)}", flush=True)
    
    # Валидация
    validation_str = model.Validate()
    if validation_str:
        print(f"Validation FAILED: {validation_str}", flush=True)
        sys.exit(1)
    else:
        print("Validation passed", flush=True)
    
    # Создаём solver
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 1
    
    print("Starting solve (num_workers=1, timeout=30s)...", flush=True)
    
    try:
        status = solver.Solve(model)
        print(f"Solve completed. Status: {solver.StatusName(status)}", flush=True)
        print(f"Time: {solver.WallTime():.2f}s", flush=True)
    except Exception as e:
        print(f"Exception during solve: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
