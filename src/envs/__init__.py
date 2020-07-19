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
    id='Letter-5x5-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv5x5'
)

register(
    id='Letter-5x5-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap5x5'
)

register(
    id='Letter-5x5-v2',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentric5x5'
)

register(
    id='Letter-5x5-v3',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMap5x5'
)

register(
    id='Letter-5x5-v4',
    entry_point='envs.gym_letters.letter_env:LetterEnvShortAgentCentric5x5'
)

register(
    id='Letter-5x5-v5',
    entry_point='envs.gym_letters.letter_env:LetterEnvShortAgentCentricFixedMap5x5'
)

register(
    id='Letter-7x7-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv7x7'
)

register(
    id='Letter-7x7-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap7x7'
)

