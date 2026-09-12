# Wait for the KDPT job (output_kdpt_v2), then RDPT, then beam-1 decode + metrics.
# Run from anywhere; paths are absolute. Requires CUDA_VISIBLE_DEVICES.

$ErrorActionPreference = "Stop"
$root = "C:\Empower\Empower"
$model = Join-Path $root "EMPOWER-MODEL"
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"

function Wait-FinetuneIdle {
    param([string]$LogPath, [string]$Marker)
    while ($true) {
        $busy = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -and $_.CommandLine -like "*finetune.py*" }
        $logHasMarker = (Test-Path $LogPath) -and (Select-String -Path $LogPath -Pattern $Marker -SimpleMatch -Quiet)
        if (-not $busy -and $logHasMarker) { return }
        Start-Sleep -Seconds 60
    }
}

Write-Host "Waiting for KDPT v2 to finish..."
Wait-FinetuneIdle -LogPath (Join-Path $root "reports\kdpt_v2_train.log") -Marker "DATALOADER:0 TEST RESULTS"
if (-not (Test-Path (Join-Path $model "output_kdpt_v2\best_pt1\config.json"))) {
    throw "KDPT finished but output_kdpt_v2/best_pt1 is missing"
}

Write-Host "Starting RDPT v3..."
Set-Location $model
& $py -u finetune.py `
    --max_source_length 128 `
    --max_knowledge_length 256 `
    --max_target_length 128 `
    --eval_max_gen_length 128 `
    --eval_min_gen_length 2 `
    --no_repeat_ngram_size 3 `
    --model_name_or_path gpt2 `
    --pfxKlgModel_name_or_path "output_kdpt_v2/best_pt1" `
    --data_dir "../data/preprocessed_v2" `
    --output_dir "output_rdpt_v3" `
    --learning_rate 3e-5 `
    --lr_scheduler linear `
    --warmup_steps 500 `
    --num_train_epochs 5 `
    --train_batch_size 8 `
    --eval_batch_size 8 `
    --n_train -1 --n_val -1 --n_test -1 `
    --tuning_mode pt2 `
    --mid_dim 800 `
    --preseqlen 64 `
    --klg_preseqlen 16 `
    --val_metric f1 `
    --save_top_k 3 `
    --early_stopping_patience 1 `
    --num_workers 0 `
    --overwrite_output_dir `
    --do_train 2>&1 | Tee-Object -FilePath (Join-Path $root "reports\rdpt_v3_train.log")

$ckpt = & $py (Join-Path $root "scripts\pick_best_ckpt.py") (Join-Path $model "output_rdpt_v3")
Write-Host "Best RDPT ckpt: $ckpt"

Write-Host "Starting decode output_gen_v2..."
& $py -u finetune.py `
    --max_source_length 128 `
    --max_knowledge_length 256 `
    --max_target_length 128 `
    --eval_max_gen_length 128 `
    --eval_min_gen_length 2 `
    --eval_beams 1 `
    --no_repeat_ngram_size 3 `
    --resumed_ckpt_file $ckpt `
    --model_name_or_path gpt2 `
    --pfxKlgModel_name_or_path "output_kdpt_v2/best_pt1" `
    --data_dir "../data/preprocessed_v2" `
    --output_dir "output_gen_v2" `
    --eval_batch_size 8 `
    --n_train -1 --n_val -1 --n_test -1 `
    --tuning_mode pt2 `
    --mid_dim 800 `
    --preseqlen 64 `
    --klg_preseqlen 16 `
    --num_workers 0 `
    --overwrite_output_dir 2>&1 | Tee-Object -FilePath (Join-Path $root "reports\gen_v2.log")

$eval = Join-Path $model "automatic_evaluation.py"
$pred = Join-Path $model "output_gen_v2\preds.txt"
$test = Join-Path $root "data\preprocessed_v2\test.json"
$out = Join-Path $root "reports\gen_v2_metrics.txt"
foreach ($m in @("f1","bleu","rouge_l","kf1")) {
    & $py $eval --pred_file $pred --test_data $test --eval_metric $m | Tee-Object -FilePath $out -Append
}
Write-Host "Wrote $out"
