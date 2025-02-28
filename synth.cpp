/* ------------------------------------------------------------
name: "synth"
Code generated with Faust 2.77.3 (https://faust.grame.fr)
Compilation options: -a minimal.cpp -lang cpp -ct 1 -es 1 -mcd 16 -mdd 1024 -mdy 33 -single -ftz 0
------------------------------------------------------------ */

#ifndef  __mydsp_H__
#define  __mydsp_H__

/************************************************************************
 IMPORTANT NOTE : this file contains two clearly delimited sections :
 the ARCHITECTURE section (in two parts) and the USER section. Each section
 is governed by its own copyright and license. Please check individually
 each section for license and copyright information.
 *************************************************************************/

/******************* BEGIN minimal.cpp ****************/
/************************************************************************
 FAUST Architecture File
 Copyright (C) 2003-2019 GRAME, Centre National de Creation Musicale
 ---------------------------------------------------------------------
 This Architecture section is free software; you can redistribute it
 and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 3 of
 the License, or (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; If not, see <http://www.gnu.org/licenses/>.
 
 EXCEPTION : As a special exception, you may create a larger work
 that contains this FAUST architecture section and distribute
 that work under terms of your choice, so long as this FAUST
 architecture section is not modified.
 
 ************************************************************************
 ************************************************************************/

#include <iostream>

#include "faust/gui/PrintUI.h"
#include "faust/gui/MapUI.h"
#ifdef LAYOUT_UI
#include "faust/gui/LayoutUI.h"
#endif
#include "faust/gui/meta.h"
#include "faust/audio/dummy-audio.h"
#ifdef FIXED_POINT
#include "faust/dsp/fixed-point.h"
#endif

// faust -a minimal.cpp noise.dsp -o noise.cpp && c++ -std=c++11 noise.cpp -o noise && ./noise

/******************************************************************************
 *******************************************************************************
 
 VECTOR INTRINSICS
 
 *******************************************************************************
 *******************************************************************************/


/********************END ARCHITECTURE SECTION (part 1/2)****************/

/**************************BEGIN USER SECTION **************************/

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif 

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <math.h>

#ifndef FAUSTCLASS 
#define FAUSTCLASS mydsp
#endif

#ifdef __APPLE__ 
#define exp10f __exp10f
#define exp10 __exp10
#endif

#if defined(_WIN32)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

class mydspSIG0 {
	
  private:
	
	int iVec0[2];
	int iRec0[2];
	
  public:
	
	int getNumInputsmydspSIG0() {
		return 0;
	}
	int getNumOutputsmydspSIG0() {
		return 1;
	}
	
	void instanceInitmydspSIG0(int sample_rate) {
		for (int l0 = 0; l0 < 2; l0 = l0 + 1) {
			iVec0[l0] = 0;
		}
		for (int l1 = 0; l1 < 2; l1 = l1 + 1) {
			iRec0[l1] = 0;
		}
	}
	
	void fillmydspSIG0(int count, float* table) {
		for (int i1 = 0; i1 < count; i1 = i1 + 1) {
			iVec0[0] = 1;
			iRec0[0] = (iVec0[1] + iRec0[1]) % 65536;
			table[i1] = std::sin(9.58738e-05f * float(iRec0[0]));
			iVec0[1] = iVec0[0];
			iRec0[1] = iRec0[0];
		}
	}

};

static mydspSIG0* newmydspSIG0() { return (mydspSIG0*)new mydspSIG0(); }
static void deletemydspSIG0(mydspSIG0* dsp) { delete dsp; }

static float ftbl0mydspSIG0[65536];

class mydsp : public dsp {
	
 private:
	
	FAUSTFLOAT fButton0;
	int iVec1[2];
	int fSampleRate;
	float fConst0;
	FAUSTFLOAT fHslider0;
	float fRec1[2];
	
 public:
	mydsp() {
	}
	
	void metadata(Meta* m) { 
		m->declare("basics.lib/name", "Faust Basic Element Library");
		m->declare("basics.lib/tabulateNd", "Copyright (C) 2023 Bart Brouns <bart@magnetophon.nl>");
		m->declare("basics.lib/version", "1.21.0");
		m->declare("compile_options", "-a minimal.cpp -lang cpp -ct 1 -es 1 -mcd 16 -mdd 1024 -mdy 33 -single -ftz 0");
		m->declare("filename", "synth.dsp");
		m->declare("maths.lib/author", "GRAME");
		m->declare("maths.lib/copyright", "GRAME");
		m->declare("maths.lib/license", "LGPL with exception");
		m->declare("maths.lib/name", "Faust Math Library");
		m->declare("maths.lib/version", "2.8.1");
		m->declare("name", "synth");
		m->declare("oscillators.lib/name", "Faust Oscillator Library");
		m->declare("oscillators.lib/version", "1.5.1");
		m->declare("platform.lib/name", "Generic Platform Library");
		m->declare("platform.lib/version", "1.3.0");
	}

	virtual int getNumInputs() {
		return 0;
	}
	virtual int getNumOutputs() {
		return 1;
	}
	
	static void classInit(int sample_rate) {
		mydspSIG0* sig0 = newmydspSIG0();
		sig0->instanceInitmydspSIG0(sample_rate);
		sig0->fillmydspSIG0(65536, ftbl0mydspSIG0);
		deletemydspSIG0(sig0);
	}
	
	virtual void instanceConstants(int sample_rate) {
		fSampleRate = sample_rate;
		fConst0 = 1.0f / std::min<float>(1.92e+05f, std::max<float>(1.0f, float(fSampleRate)));
	}
	
	virtual void instanceResetUserInterface() {
		fButton0 = FAUSTFLOAT(0.0f);
		fHslider0 = FAUSTFLOAT(4.4e+02f);
	}
	
	virtual void instanceClear() {
		for (int l2 = 0; l2 < 2; l2 = l2 + 1) {
			iVec1[l2] = 0;
		}
		for (int l3 = 0; l3 < 2; l3 = l3 + 1) {
			fRec1[l3] = 0.0f;
		}
	}
	
	virtual void init(int sample_rate) {
		classInit(sample_rate);
		instanceInit(sample_rate);
	}
	
	virtual void instanceInit(int sample_rate) {
		instanceConstants(sample_rate);
		instanceResetUserInterface();
		instanceClear();
	}
	
	virtual mydsp* clone() {
		return new mydsp();
	}
	
	virtual int getSampleRate() {
		return fSampleRate;
	}
	
	virtual void buildUserInterface(UI* ui_interface) {
		ui_interface->openVerticalBox("synth");
		ui_interface->addHorizontalSlider("freq", &fHslider0, FAUSTFLOAT(4.4e+02f), FAUSTFLOAT(2e+01f), FAUSTFLOAT(2e+04f), FAUSTFLOAT(0.01f));
		ui_interface->addButton("gate", &fButton0);
		ui_interface->closeBox();
	}
	
	virtual void compute(int count, FAUSTFLOAT** RESTRICT inputs, FAUSTFLOAT** RESTRICT outputs) {
		FAUSTFLOAT* output0 = outputs[0];
		float fSlow0 = float(fButton0);
		float fSlow1 = fConst0 * float(fHslider0);
		for (int i0 = 0; i0 < count; i0 = i0 + 1) {
			iVec1[0] = 1;
			float fTemp0 = ((1 - iVec1[1]) ? 0.0f : fSlow1 + fRec1[1]);
			fRec1[0] = fTemp0 - std::floor(fTemp0);
			output0[i0] = FAUSTFLOAT(fSlow0 * ftbl0mydspSIG0[std::max<int>(0, std::min<int>(int(65536.0f * fRec1[0]), 65535))]);
			iVec1[1] = iVec1[0];
			fRec1[1] = fRec1[0];
		}
	}

};

/***************************END USER SECTION ***************************/

/*******************BEGIN ARCHITECTURE SECTION (part 2/2)***************/

using namespace std;

#ifdef LAYOUT_UI
void getMinimumSize(dsp* dsp, LayoutUI* ui, float& width, float& height)
{
    // Prepare layout
    dsp->buildUserInterface(ui);
    
    cout << "==========================" << endl;
    for (const auto& it : ui->fPathItemMap) {
        cout << it.second << endl;
    }
    
    cout << "Width " << ui->getWidth() << endl;
    cout << "Height " << ui->getHeight() << endl;
    
    width = ui->getWidth();
    height = ui->getHeight();
}

void setPosAndSize(LayoutUI* ui, float x_pos, float y_pos, float width, float height)
{
    ui->setSize(width, height);
    ui->setPos(x_pos, y_pos);
    
    cout << "==========================" << endl;
    for (const auto& it : ui->fPathItemMap) {
        cout << it.second << endl;
    }
    
    cout << "Width " << ui->getWidth() << endl;
    cout << "Height " << ui->getHeight() << endl;
}
#endif

int main(int argc, char* argv[])
{
    mydsp DSP;
    cout << "DSP size: " << sizeof(DSP) << " bytes\n";
    
    // Activate the UI, here that only print the control paths
    PrintUI print_ui;
    DSP.buildUserInterface(&print_ui);
    
    /*
    MapUI map_ui;
    DSP.buildUserInterface(&map_ui);
    for (int i = 0; i < map_ui.getParamsCount(); i++) {
        cout << "getParamAddress " << map_ui.getParamAddress(i) << endl;
        cout << "getParamShortname " << map_ui.getParamShortname(i) << endl;
        cout << "getParamLabel " << map_ui.getParamLabel(i) << endl;
    }
    */
    
#ifdef LAYOUT_UI
    LayoutUI layout_ui;
    float width, height;
    getMinimumSize(&DSP, &layout_ui, width, height);
    cout << "minimal_width: " << width << "\n";
    cout << "minimal_height: " << height << "\n";
    setPosAndSize(&layout_ui, 0, 0, width*1.5, height*1.5);
#else
    // Allocate the audio driver to render 5 buffers of 512 frames
    dummyaudio audio(5);
    audio.init("Test", static_cast<dsp*>(&DSP));
    
    // Render buffers...
    audio.start();
    audio.stop();
#endif
}

/******************* END minimal.cpp ****************/


#endif
