from gym.envs.registration import register


register(
    id='Letter-4x4-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv4x4'
)
