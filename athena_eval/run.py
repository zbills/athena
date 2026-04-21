"""Run models against benchmark tasks and store predictions."""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm
import concurrent.futures 
from .utils import load_jsonl, load_yaml
from .answer_extractors import extract_answer
from .models import load_model

from .evaluate import load_alias_dict, load_related_dict, score_record

import pandas as pd
def existing_ids(path: Path) -> set[int]:
    """Return a set of record identifiers already stored in *path*."""
    ids = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ids.add(obj.get("id"))
                except Exception:
                    continue
    return ids


def process_record(  
    idx: int,  
    row: Dict,
    task_name: str,  
    alias_dict,  
    related_dict, 
    model, 
    evaluate: bool  
) -> tuple:  
    """  
    Process one record: format prompt, call the model using the global MODEL,  
    extract the prediction and, if evaluation is enabled, score it.  
  
    Returns:  
        A tuple containing:  
          - rec: a dict with keys: id, prompt, response, prediction, answer  
          - score: the returned score (or None)  
          - success: a boolean indicating whether the record was successfully processed.  
    """  
    # Modify the prompt as in the original code.  
    prompt = row.get("prompt", "").replace(  
        "Briefly justify your choice.",  
        "Please directly respond with the answer without explanation"  
    ).strip()  
    answer = row.get("answer", "")  
      
    # Use the global MODEL loaded in the worker.  
    response = model.generate(prompt, answer=answer)  
    pred = extract_answer(task_name, response)  
    rec = {  
        "id": idx,  
        "prompt": prompt,  
        "response": response,  
        "prediction": pred,  
        "answer": answer,  
    }  
      
    if evaluate:  
        score, success = score_record(task_name, pred, answer, alias_dict, related_dict)  
        rec["score"] = score  
        rec["success"] = success  
        return rec, score, success  
    else:  
        return rec, None, None  

def run_model_on_task(  
    model_cfg: Dict,  
    task_name: str,  
    round: int,  
    dataset_path: str,  
    out_dir: Path,  
    evaluate: bool = False,  
    alias_dict=None,  
    related_dict=None,  
) -> Dict[str, float] | None:  
    """  
    Generate predictions for a dataset and optionally score them.  
      
    Parameters  
    ----------  
    model_cfg:  
        Configuration dictionary for the model.  
    task_name:  
        Name of the benchmark task.  
    dataset_path:  
        Path to the dataset file.  
    out_dir:  
        Directory where prediction files are written.  
    evaluate:  
        If ``True``, compute running scores and save them alongside predictions.  
    alias_dict, related_dict:  
        Lookup tables used for the TAA task.  
      
    Returns  
    -------  
    dict | None  
        Aggregate metrics if ``evaluate`` is ``True``; otherwise ``None``.  
    """  
    model_name = model_cfg.get("name") or model_cfg.get("model")  
    model_dir = out_dir / f"{model_name}_{round}"  
    preds_path = model_dir / f"{task_name}.jsonl"
    model = load_model(model_cfg)  
    done = existing_ids(preds_path)  
    records = load_jsonl(dataset_path)  
    model_dir.mkdir(parents=True, exist_ok=True)  
      
    # Running aggregates for progress display.  
    sum_score = 0.0  
    sum_correct = 0  
    sum_plausible = 0  
    sum_combined = 0.0  
    count_success = 0  
  
    print(f"[run_model_on_task] Running model {model_name} on task {task_name}, round {round}, total records: {len(records)}, already done: {len(done)}")  
      
    # Make a list of (idx, row) for records that have not been processed already.  
    tasks_to_process = [(idx, row) for idx, row in enumerate(records) if idx not in done]  
    total_tasks = len(tasks_to_process)  
  
    # Open the file for appending output.  
    with preds_path.open("a", encoding="utf-8") as f, concurrent.futures.ProcessPoolExecutor(max_workers=model_cfg.get("workers", 10)) as executor, tqdm(total=total_tasks, desc=f"{model_name}-{task_name}") as pbar:  
          
        futures = {}  
        task_iter = iter(tasks_to_process)  
          
        # Submit an initial batch of up to 10 tasks.  
        for _ in range(model_cfg.get("workers", 10)):  
            try:  
                idx, row = next(task_iter)  
                future = executor.submit(  
                    process_record, idx, row, task_name, alias_dict, related_dict, model, evaluate  
                )  
                futures[future] = idx  
            except StopIteration:  
                break  
          
        # Process futures as they complete and keep submitting new tasks.  
        while futures:  
            # Wait for at least one future to finish.  
            done_set, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)  
            for future in done_set:  
                rec, score, success = future.result()  
               
                # Write the record JSON to output.  
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")  
                  
                if evaluate and success:  
                    count_success += 1  
                    if isinstance(score, dict):  
                        sum_correct += score.get("correct", 0)  
                        sum_plausible += score.get("plausible", 0)  
                        sum_combined += score.get("combined", 0.0)  
                        pbar.set_postfix({  
                            "avg_correct": f"{sum_correct / count_success:.3f}",  
                            "avg_plausible": f"{sum_plausible / count_success:.3f}",  
                            "avg_combined": f"{sum_combined / count_success:.3f}",  
                        })  
                    else:  
                        sum_score += float(score)  
                        pbar.set_postfix({  
                             "avg_score": f"{sum_score / count_success:.3f}"  
                        })  
                pbar.update(1)  
                  
                # Remove the finished future.  
                del futures[future]  
                  
                # Submit a new task (if any remain) so that we maintain 10 in-flight.  
                try:  
                    idx, row = next(task_iter)  
                    new_future = executor.submit(  
                        process_record, idx, row, task_name, alias_dict, related_dict, model,evaluate  
                    )  
                    futures[new_future] = idx  
                except StopIteration:  
                    # No more tasks to submit.  
                    pass  
  
    # Return aggregate results if evaluation was performed.  
    if not evaluate or count_success == 0:  
         return None  
  
    task_upper = task_name.upper()  
    if task_upper == "TAA":  
        return {  
            "correct_accuracy": sum_correct / count_success,  
            "plausible_accuracy": sum_plausible / count_success,  
            "combined_accuracy": sum_combined / count_success,  
        }  
    if task_upper == "RMS":  
        return {"f1": sum_score / count_success}  
    metric_name = "mean_absolute_deviation" if task_upper == "CVSS" else "accuracy"  
    return {  
         "task_name": task_name,  
         "metric_name": metric_name,  
         "metric_value": sum_score / count_success,  
         "sum_correct": sum_correct,  
         "count_success": count_success,  
    }  


# Define the worker function at the top level of the module.  
def run_round(r, model_cfg, t, dataset_path, pred_dir, evaluate, alias_dict, related_dict):  
    metrics = run_model_on_task(  
        model_cfg,  
        t,  
        r,  
        dataset_path,  
        pred_dir,  
        evaluate=evaluate,  
        alias_dict=alias_dict,  
        related_dict=related_dict,  
    )  
    return r, metrics  

def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point to run models on benchmark tasks."""
    print("[run] Starting main function")
    parser = argparse.ArgumentParser(description="Run models on benchmark tasks")
    parser.add_argument("--config", default="athena_eval/config.yaml")
    parser.add_argument("--model", help="model name to run", default=None)
    parser.add_argument("--task", help="task to run", default=None)
    parser.add_argument("--rounds", help="rounds to run", default=3)
    parser.add_argument("--workers", help="number of workers", default=10, type=int)
    parser.add_argument("--endpoint_url", help="OpenAI endpoint URL", default=None)
    parser.add_argument("--use_proxy", help="Use OpenAI proxy model", action="store_true", default=False)

    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=True,
        help="evaluate predictions during generation (default)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="skip evaluation step",
    )
    parser.add_argument(
        "--mini",
        dest="mini",
        action="store_true",
        default=False,
        help="run only on mini benchmark subsets; write outputs to runs-mini",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_yaml(args.config)
    pred_dir = Path("runs-mini") if args.mini else Path(cfg.get("predictions_dir", "data/predictions"))

    alias_dict = related_dict = None
    if args.evaluate:
        base_dir = Path(__file__).resolve().parent
        alias_csv = base_dir / "taa" / "aliases.csv"
        related_csv = base_dir / "taa" / "related_groups.csv"
        alias_dict = load_alias_dict(str(alias_csv))
        related_dict = load_related_dict(str(related_csv))

    model_names = [args.model] if args.model else list(cfg.get("models", {}).keys())
    tasks_cfg = cfg.get("tasks", {})

    def resolve_task_name(name: str) -> str:
        alias_map = {"MCQ3K": "CKT", "MCQ": "CKT"}
        upper = name.upper()
        if upper in alias_map:
            return alias_map[upper]
        for candidate in tasks_cfg.keys():
            if candidate.upper() == upper:
                return candidate
        return name

    # Assuming task_names, model_names, cfg, args, pred_dir, alias_dict, and related_dict are defined upstream.  
    # For example, task_names might be defined as:  
    # task_names = [resolve_task_name(args.task)] if args.task else list(tasks_cfg.keys())  
    
    records = []  
    task_names = [resolve_task_name(args.task)] if args.task else list(tasks_cfg.keys())
    # Construct a list of all tasks that need to be done.  
    all_tasks = []  
    for m in model_names:  
        model_cfg = cfg["models"][m]  
        model_name = model_cfg.get("name") or model_cfg.get("model")  
        endpoint_url = args.endpoint_url or model_cfg.get("endpoint_url")
        model_cfg["endpoint_url"] = endpoint_url  # Ensure the model config has the endpoint
        model_cfg['workers'] = args.workers
        model_cfg['use_proxy'] = args.use_proxy

        for t in task_names:  
            dataset_path = cfg["tasks"][t]  
            # If --mini, point to benchmark-mini/<basename> when available.  
            if args.mini:  
                mini_path = Path("benchmark-mini") / Path(dataset_path).name  
                if mini_path.exists():  
                    dataset_path = str(mini_path)  
                else:  
                    print(f"[run] Mini dataset missing for {t} at {mini_path}; using full dataset {dataset_path}")  
            for r in range(int(args.rounds)):  
                # Each task is represented as a tuple containing all needed parameters.  
                all_tasks.append(  
                    (r, model_cfg, t, dataset_path, pred_dir, args.evaluate, alias_dict, related_dict, model_name)  
                )  
    
    # Use a ProcessPoolExecutor with max_workers=10. 
    # we are running records in the task in parallel now so here we just submit the task one by one 
    max_workers_count = 1 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_count) as executor:  
        future_to_task = {}  
        #print(f"[run] Submitting up to {max_workers_count} initial tasks, all tasks: {all_tasks} ")
        # Submit the first batch of tasks (up to 1).  
        initial_task_count = min(max_workers_count, len(all_tasks))  
        for i in range(initial_task_count):  
            task = all_tasks[i]  
            r, model_cfg, t, dataset_path, pred_dir, evaluate, alias_dict, related_dict, model_name = task  
            print(f"[run] Running model {model_name} on task {t}, round {r}")  
            future = executor.submit(  
                run_round, r, model_cfg, t, dataset_path, pred_dir, evaluate, alias_dict, related_dict  
            )  
            future_to_task[future] = task  
    
        next_task_index = initial_task_count  
    
        # Process futures as they complete.  
        while future_to_task:  
            # Wait until at least one running task finishes.  
            done, _ = concurrent.futures.wait(  
                future_to_task, return_when=concurrent.futures.FIRST_COMPLETED  
            )  
            for future in done:  
                task_info = future_to_task[future]  
                # Remove the finished future from our tracking dict.  
                del future_to_task[future]  
                # Assume run_round returns a tuple (round, metrics).  
                r, metrics = future.result()  
                # We stored model_name and task name (t) in task_info for logging.  
                _, _, t, _, _, _, _, _, model_name = task_info  
                if metrics is not None:  
                    print(f"{model_name} {t}: {metrics}")  
                    records.append(metrics)  
                    
                # Submit a new task if any remain.  
                if next_task_index < len(all_tasks):  
                    new_task = all_tasks[next_task_index]  
                    next_task_index += 1  
    
                    r_new, model_cfg_new, t_new, dataset_path_new, pred_dir_new, evaluate_new, alias_dict_new, related_dict_new, model_name_new = new_task  
                    print(f"[run] Running model {model_name_new} on task {t_new}, round {r_new}")  
                    new_future = executor.submit(  
                        run_round, r_new, model_cfg_new, t_new, dataset_path_new, pred_dir_new, evaluate_new, alias_dict_new, related_dict_new  
                    )  
                    future_to_task[new_future] = new_task  
    pd.DataFrame.from_records(records).to_csv(base_dir / f"{model_name}.csv")


if __name__ == "__main__":  # pragma: no cover
    print("[run] Executing main")
    main()
