document.addEventListener("DOMContentLoaded", async () => {
    if (!navigator.requestMIDIAccess) {
        console.log("Web MIDI API non supportée");
        return;
    }

    try {
        const midiAccess = await navigator.requestMIDIAccess();
        const inputs = Array.from(midiAccess.inputs.values());
        const outputs = Array.from(midiAccess.outputs.values());

        if (inputs.length === 0 || outputs.length === 0) {
            console.log("Aucune entrée ou sortie MIDI détectée");
            return;
        }

        const input = inputs[1]; // Première entrée MIDI
        const output = outputs[0]; // Première sortie MIDI

        console.log(`Utilisation de l'entrée: ${input.name}`);
        console.log(`Utilisation de la sortie: ${output.name}`);

        input.onmidimessage = (event) => {
            const [status, note, velocity] = event.data;
            if (status >= 128 && status < 144) { // Note Off
                output.send([status, note + 2, velocity]);
            } else if (status >= 144 && status < 160) { // Note On
                output.send([status, note + 2, velocity]);
            }
        };
    } catch (error) {
        console.error("Erreur d'accès au MIDI: ", error);
    }
});
