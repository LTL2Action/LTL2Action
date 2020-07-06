from gym.envs.registration import register


register(
    id='Letter-4x4-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv4x4'
)

register(
    id='Letter-4x4-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap4x4'
)

register(
    id='Letter-7x7-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv7x7'
)

register(
    id='Letter-7x7-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap7x7'
)

