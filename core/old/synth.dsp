import("stdfaust.lib");

freq = hslider("freq", 440, 20, 20000, 0.01);
gate = button("gate");

process = os.osc(freq) * gate;
