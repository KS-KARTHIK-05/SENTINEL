# Inter-UAV broadcast. The payload is a serialised sentinel_core.agent.Beacon;
# all traffic passes through comm_emulator, which applies the channel model.
uint8 sender
uint8 receiver            # set by comm_emulator on delivery (255 = not yet routed)
float64 stamp
uint8[] payload
