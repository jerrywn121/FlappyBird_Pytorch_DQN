#!/bin/bash
if [ -d scores ]
then
    echo 'directory scores exists!'
    exit 1
fi

#cmd="nohup python -u main.py --algo QLearning --epsilon 0.1 --rounding 5 --lr 0.1 --numTestIters 20 > ql_table.log 2>&1 &"
# cmd="nohup python -u main.py --algo FuncApproxDNN --epsilon 0.1 --lr 0.0001 --numTestIters 20 > dnn.log 2>&1 &"
cmd="nohup python -u main.py --algo FuncApproxCNN --epsilon 0.1 --lr 0.0001 --numTestIters 20 --past_frame 20 > cnn_frame_20.log 2>&1 &"
# cmd="nohup python -u main.py --algo FuncApproxRNN --epsilon 0.1 --lr 0.0001 --numTestIters 20 > lstm.log 2>&1 &"
echo ${cmd}
eval ${cmd}

echo $! > save_pid.txt
tail -f *.log
