__all__ = [
    'calculate_detected_cases'
]


def calculate_detected_cases(infected_asymptomatic,
                             infected_mild,
                             infected_severe,
                             removed_asymptomatic,
                             removed_mild,
                             removed_severe,
                             ratio_asymptomatic_detected: float = 0.,
                             ratio_mild_detected: float = 0.3,
                             ratio_severe_detected: float = 1.):

    out = ratio_asymptomatic_detected * (infected_asymptomatic + removed_asymptomatic) \
          + ratio_mild_detected * (infected_mild + removed_mild) \
          + ratio_severe_detected * (infected_severe * removed_severe)

    return out