/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

 This program is free software: you can use its results,
 redistribute it and/or modify it under the terms of the GNU
 Affero General Public License version 3 as published by the
 Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program ('agpl-3.0.txt').
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#ifndef UNIT_TESTING

#ifndef H_CONTROL_PANEL
#define H_CONTROL_PANEL

#include <opencv2/core.hpp>

// forward declarations
class Settings;
struct IControlPanelActions;
class ImgSettings;
class MatchSettings;

/**
Configures the transformation settings, chooses which image to process and with which cmap.

The range of the slider used for selecting the encoding is updated automatically
based on the selected font family.

The sliders from OpenCV are quite basic, so ranges must be integer, start from 0 and include 1.
Thus, range 0..1 could also mean there's only one value (0) and the 1 will be ignored.
Furthermore, range 0..x can be invalid if the actual range is [x+1 .. y].
In this latter case an error message will report the problem and the user needs to correct it.
*/
class ControlPanel {
protected:
	/**
	Helper to convert settings from actual ranges to the ones used by the sliders.

	For now, flexibility in choosing the conversion rules is more important than design,
	so lots of redundancies appear below.
	*/
	struct Converter {
		/**
		One possible conversion function 'proportionRule':
		x in 0..xMax;  y in 0..yMax  =>
		y = x*yMax/xMax
		*/
		static double proportionRule(double x, double xMax, double yMax);

		/// used for the slider controlling the structural similarity
		struct StructuralSim {
			static int toSlider(double ssim);
			static double fromSlider(int ssim);
		};

		/// used for the 3 sliders controlling the correctness
		struct Correctness {
			static int toSlider(double correctness);
			static double fromSlider(int correctness);
		};

		/// used for the slider controlling the contrast
		struct Contrast {
			static int toSlider(double contrast);
			static double fromSlider(int contrast);
		};

		/// used for the slider controlling the 'gravitational' smoothness
		struct Gravity {
			static int toSlider(double gravity);
			static double fromSlider(int gravity);
		};

		/// used for the slider controlling the directional smoothness
		struct Direction {
			static int toSlider(double direction);
			static double fromSlider(int direction);
		};

		/// used for the slider controlling the preference for larger symbols
		struct LargerSym {
			static int toSlider(double largerSym);
			static double fromSlider(int largerSym);
		};
	};

	IControlPanelActions &actions;	// window manager

	// Configuration sliders' positions
	int maxHSyms, maxVSyms;
	int encoding, fontSz;
	int symsBatchSz;
	int hybridResult;
	int structuralSim, underGlyphCorrectness, glyphEdgeCorrectness, asideGlyphCorrectness;
	int moreContrast, gravity, direction, largerSym, thresh4Blanks;

	/**
	hack field

	The max of the encodings slider won't update unless
	issuing an additional slider move, which has to be ignored.
	*/
	bool updatingEncMax = false;

public:
	ControlPanel(IControlPanelActions &actions_, const Settings &cfg);

	void updateEncodingsCount(unsigned uniqueEncodings);	///< puts also the slider on 0
	bool encMaxHack() const { return updatingEncMax; }		///< used for the hack above

	/// updates font size & encoding sliders, if necessary
	void updateSymSettings(unsigned encIdx, unsigned fontSz_);
	void updateImgSettings(const ImgSettings &is); ///< updates sliders concerning ImgSettings items
	void updateMatchSettings(const MatchSettings &ms); ///< updates sliders concerning MatchSettings items
};

#endif // H_CONTROL_PANEL

#endif // UNIT_TESTING
