document.addEventListener("DOMContentLoaded", async () => {
            if (!navigator.requestMIDIAccess) {
                console.log("Web MIDI API non supportée");
                return;
            }
    try {
        const midiAccess = await navigator.requestMIDIAccess();
        let inputs = Array.from(midiAccess.inputs.values());
        let outputs = Array.from(midiAccess.outputs.values());

        if (inputs.length === 0 || outputs.length === 0) {
            console.log("Aucune entrée ou sortie MIDI détectée");
            return;
        }

        let input = inputs[0]; // Première entrée MIDI par défaut
        let output = outputs[0]; // Première sortie MIDI par défaut

       function updateMidiDevices() {
            inputs = Array.from(midiAccess.inputs.values());
            outputs = Array.from(midiAccess.outputs.values());
        }

        function updateMidiInput(selectedIndex) {
            updateMidiDevices();
            if (inputs[selectedIndex]) {
                if (input) {
                    input.onmidimessage = null; // Supprimer l'ancien écouteur
                }
                input = inputs[selectedIndex];
                console.log(`Nouvelle entrée MIDI sélectionnée: ${input.name}`);
                input.onmidimessage = handleMidiMessage;
            }
        }

        function updateMidiOutput(selectedIndex) {
            updateMidiDevices();
            if (outputs[selectedIndex]) {
                output = outputs[selectedIndex];
                console.log(`Nouvelle sortie MIDI sélectionnée: ${output.name}`);
            }
        }

        let melody = [];
        let activeNotes = new Set();
        let playedNotes = new Map(); // Utilisation d'une Map pour suivre les notes jouées et leur statut
        let lastNoteTime = 0;
        const silenceThreshold = 1000; // 1 seconde en ms
        let playbackTimeout = null;
        let playbackActive = false;
        let playbackTimers = [];

        function handleMidiMessage(event) {
            console.log("MIDI reçu :", event.data); // Debugging
            if (!output) {
                console.warn("Aucune sortie MIDI sélectionnée");
                return;
            }
            const [status, note, velocity] = event.data;
            const currentTime = performance.now();

            if (playbackTimeout) {
                clearTimeout(playbackTimeout);
            }

            if (playbackActive) {
                stopPlayback();
            }

            if (status >= 144 && status < 160 && velocity > 0) { // Note On
                activeNotes.add(note);
                melody.push([status, note, velocity, currentTime]);
            } else if (status >= 128 && status < 160) { // Note Off
                activeNotes.delete(note);
                melody.push([status, note, velocity, currentTime]);
            }

            lastNoteTime = currentTime;

            playbackTimeout = setTimeout(checkAndPlayMelody, silenceThreshold);
        };

        input.onmidimessage = handleMidiMessage;

        function checkAndPlayMelody() {
            if (melody.length === 0 || activeNotes.size > 0) return;
            playTransposedMelody();
        }

        function playTransposedMelody() {
            console.log("Rejoue la mélodie transposée...");
            const startTime = melody[0][3];
            playbackActive = true;
            playedNotes.clear();

            melody.forEach(([status, note, velocity, time]) => {
                const delay = time - startTime;
                const transposedNote = note + 2;

                const timer = setTimeout(() => {
                    output.send([status, transposedNote, velocity]);
                    if (status >= 144 && status < 160 && velocity > 0) {
                        playedNotes.set(transposedNote, status); // Stocker la note jouée avec son statut
                    } else if (status >= 128 && status < 160) {
                        playedNotes.delete(transposedNote); // Retirer la note si elle est arrêtée
                    }
                }, delay);
                playbackTimers.push(timer);
            });
            melody = []
        }

        function stopPlayback() {
            console.log("Arrêt de la lecture en cours");
            playbackTimers.forEach(clearTimeout);
            playbackTimers = [];
            playbackActive = false;

            // Envoyer les Note Off pour toutes les notes jouées qui n'ont pas encore été arrêtées
            playedNotes.forEach((status, note) => {
                if (status >= 144 && status < 160) { // Si une Note On est active
                    output.send([128, note, 0]); // Envoyer un Note Off
                }
            });
            playedNotes.clear();
        }
   function updateMidiInput(selectedIndex) {
            console.log("updateMidiInput");

            if (inputs[selectedIndex]) {
                if (input) {
                    input.onmidimessage = null; // Supprimer l'ancien écouteur
                }
                input = inputs[selectedIndex];
                console.log(`Nouvelle entrée MIDI sélectionnée: ${input.name}`);
                input.onmidimessage = handleMidiMessage;
            }
        }

        function updateMidiOutput(selectedIndex) {
            if (outputs[selectedIndex]) {
                output = outputs[selectedIndex];
                console.log(`Nouvelle sortie MIDI sélectionnée: ${output.name}`);
            }
        }
    } catch (error) {
        console.error("Erreur d'accès au MIDI: ", error);
    }

      // Rendre ces fonctions accessibles globalement pour l'appel depuis index.html
        window.handleMidiMessage = handleMidiMessage;
        window.updateMidiInput = updateMidiInput;
        window.updateMidiOutput = updateMidiOutput;

});