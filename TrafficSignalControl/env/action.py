from traci import trafficlight


def action(tls_id, phase, duration):
    """
    Change the phase of the traffic light with the given id to the given phase
    for the given duration.

    :param tls_id: The id of the traffic light to change the phase of.
    :param phase: The phase to change the traffic light to.
    :param duration: The duration of the phase change.
    :return: None
    """
    trafficlight.setPhase(tls_id, phase)
    trafficlight.setPhaseDuration(tls_id, duration)

