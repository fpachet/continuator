import mido
from mido import Message

# msg = mido.Message('note_on', note=60)
# msg.copy(channel=2)
# Message('note_on', channel=2, note=60, velocity=64, time=0)
# port = mido.open_output('Port Name')
# port.send(msg)

print(mido.version)
inports = mido.get_input_names()
outports = mido.get_output_names()
print(inports)
print(outports)

msg = Message('note_on', note=61)
msg2 = Message('note_off', note=61, velocity=64, time=5000)

outport = mido.open_output()
outport.send(msg)
outport.send(msg2)
print(outport)