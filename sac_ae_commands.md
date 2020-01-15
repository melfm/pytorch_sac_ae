# Half-Cheetah
################################

# TODO: Its not optional atm
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_Only \
    --expert_dir ./half_cheetah_expert \
    --bc_learning
    --seed 1


python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_RL \
    --expert_dir ./half_cheetah_expert \
    --bc_learning
    --seed 1


python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_RL_with_noise \
    --expert_dir ./half_cheetah_expert \
    --bc_learning
    --demo_noise \
    --seed 1



python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_RL_Q_Filter \
    --expert_dir ./half_cheetah_expert \
    --bc_learning \
    --q_filter \
    --seed 1


python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_RL_Q_Filter_with_noise \
    --expert_dir ./half_cheetah_expert \
    --bc_learning \
    --q_filter \
    --demo_noise \
    --seed 1



# TODO:
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cheetah_BC_RL_Shaping \
    --expert_dir ./half_cheetah_expert \
    --bc_learning
    --seed 1


################################
# Cartpole
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --save_model \
    --work_dir ./cartpole_bc_separate_buffers \
    --expert_dir ./cartpole_swingup_expert \
    --bc_learning
    --seed 1




