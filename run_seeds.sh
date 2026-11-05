for s in 1 2; do
  for r in none time hack fix; do
    if [ -f models/${r}_s$s.zip ]; then
      echo "skip ${r}_s$s (already done)"
    else
      python train.py --reward $r --name ${r}_s$s --seed $s
    fi
  done
done