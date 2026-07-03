$seeds = 0, 1, 2
$dims = 64, 128

foreach ($dim in $dims) {
    foreach ($seed in $seeds) {
        $logFile = "training_logs_3L_Rank_05_LowLR_dim${dim}_seed${seed}.txt"
        Write-Host "Running Dim=$dim Seed=$seed..."
        python -u models\train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim $dim --seed $seed > $logFile 2>&1
    }
}
Write-Host "All seeded runs complete."
