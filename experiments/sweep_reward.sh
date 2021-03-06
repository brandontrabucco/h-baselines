for pre_exp_reward_scale in 1.0 2.0 10.0; do
    for pre_exp_reward_shift in 0.5 0.0 -0.5; do
        nohup python run_hrl.py "HumanoidMaze" \
            --total_steps 200000 \
            --relative_goals \
            --use_huber \
            --save_interval 100000 \
            --connected_gradients \
            --cg_weights 0.001 \
            --initial_exploration_steps 25000 \
            --buffer_size 50000 \
            --batch_size 128 \
            --pre_exp_reward_scale $pre_exp_reward_scale \
            --pre_exp_reward_shift $pre_exp_reward_shift &
        sleep 5
    done
done


python run_hrl.py "ImageAntMaze" \
  --total_steps 5000000 \
  --relative_goals \
  --use_huber \
  --save_interval 100000 \
  --cooperative_gradients \
  --cg_weights 0.01 \
  --initial_exploration_steps 10000 \
  --buffer_size 100000 \
  --batch_size 128 \
  --model_params:model_type conv \
  --model_params:image_height 32 \
  --model_params:image_width 32 \
  --model_params:image_channels 3 \
