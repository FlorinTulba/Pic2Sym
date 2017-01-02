/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#ifndef UNIT_TESTING

#ifndef H_CONTROL_PANEL
#define H_CONTROL_PANEL

#include "appState.h"

#pragma warning ( push, 0 )

#include <set>
#include <map>
#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
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

	IControlPanelActions &performer;	///< the delegate responsible to perform selected actions
	const Settings &cfg;				///< the settings, required to (re)initialize the sliders

	/// pointers to the names of the sliders that are undergoing value restoration
	std::set<const cv::String> slidersRestoringValue;

	/**
	When performing Load All Settings or only Load Match Aspects Settings, the corresponding sliders
	need to be updated one by one without considering the state and without modifying this state.
	In order to reduce the chance that some parallel update setting event might get also a free ride,
	the sliders are authorized one by one.
	*/
	const cv::String *pLuckySliderName = nullptr;

	/**
	Application state manipulated via actionDemand() method
	and the ActionPermit-s generated by that method.
	Its inspection & update are guarded by a lock mechanism.
	*/
	volatile AppStateType appState = ST(Idle);

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
	ControlPanel(IControlPanelActions &performer_, const Settings &cfg_);
	ControlPanel(const ControlPanel&) = delete;
	ControlPanel(ControlPanel&&) = delete;
	void operator=(const ControlPanel&) = delete;
	void operator=(ControlPanel&&) = delete;

	/**
	Restores a slider to its previous value when:
	- the newly set value was invalid
	- the update of the tracker happened while the application was not in the appropriate state for it

	Only when entering the explicit new desired value of the slider there will be a single
	setTrackPos event.

	Otherwise (when dragging the handle of a slider, or when clicking on a new target slider position),
	there will be several intermediary positions that generate setTrackPos events.

	So, if changing the trackbar is requested at an inappropriate moment,
	all intermediary setTrackPos events should be discarded.
	This can be accomplished as follows:
	- including trName in slidersRestoringValue during the handling of the first setTrackPos event
	  (ensures trName won't be allowed to change the corresponding value from the settings until
	  trName is removed from slidersRestoringValue)
	- starting a thread that:
		- displays a modal window with the error text, thus blocking any further user maneuvers
		- after the user closes the mentioned modal window, all the setTrackPos events should have
		  been consumed and now the restoration of the previous value can finally proceed
		  at the termination of this thread
	*/
	void restoreSliderValue(const cv::String &trName, const std::string &errText);

	/// Authorizes the action of the control whose name is provided as parameter.
	/// When the action isn't allowed returns nullptr.
	std::unique_ptr<ActionPermit> actionDemand(const cv::String &controlName);

	void updateEncodingsCount(unsigned uniqueEncodings);	///< puts also the slider on 0
	bool encMaxHack() const { return updatingEncMax; }		///< used for the hack above

	/// updates font size & encoding sliders, if necessary
	void updateSymSettings(unsigned encIdx, unsigned fontSz_);
	void updateImgSettings(const ImgSettings &is); ///< updates sliders concerning ImgSettings items
	void updateMatchSettings(const MatchSettings &ms); ///< updates sliders concerning MatchSettings items
};

#endif // H_CONTROL_PANEL

#endif // UNIT_TESTING
