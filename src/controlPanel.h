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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#ifdef UNIT_TESTING
#include "../test/mockUi.h"

#else // UNIT_TESTING not defined

#ifndef H_CONTROL_PANEL
#define H_CONTROL_PANEL

#include "appState.h"

#pragma warning ( push, 0 )

#include "std_string.h"
#include "std_memory.h"
#include <unordered_set>
#include <unordered_map>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
struct ISettings;
struct IControlPanelActions;
struct IfImgSettings;
struct IMatchSettings;
class SliderConverter;

/// Interface of ControlPanel
struct IControlPanel /*abstract*/ {
	virtual void restoreSliderValue(const cv::String &trName, const std::stringType &errText) = 0;

	/// Authorizes the action of the control whose name is provided as parameter.
	/// When the action isn't allowed returns nullptr.
	virtual std::uniquePtr<const ActionPermit> actionDemand(const cv::String &controlName) = 0;

	virtual void updateEncodingsCount(unsigned uniqueEncodings) = 0;	///< puts also the slider on 0
	virtual bool encMaxHack() const = 0;		///< used for the hack above

	/// updates font size & encoding sliders, if necessary
	virtual void updateSymSettings(unsigned encIdx, unsigned fontSz_) = 0;
	virtual void updateImgSettings(const IfImgSettings &is) = 0; ///< updates sliders concerning IfImgSettings items
	virtual void updateMatchSettings(const IMatchSettings &ms) = 0; ///< updates sliders concerning IMatchSettings items

	virtual ~IControlPanel() = 0 {}
};

/**
Configures the transformation settings, chooses which image to process and with which cmap.

The range of the slider used for selecting the encoding is updated automatically
based on the selected font family.

The sliders from OpenCV are quite basic, so ranges must be integer, start from 0 and include 1.
Thus, range 0..1 could also mean there's only one value (0) and the 1 will be ignored.
Furthermore, range 0..x can be invalid if the actual range is [x+1 .. y].
In this latter case an error message will report the problem and the user needs to correct it.
*/
class ControlPanel : public IControlPanel {
protected:
	/**
	Map between:
	- the addresses of the names of the sliders corresponding to the matching aspects
	- the slider to/from value converter for each such slider
	*/
	static const std::unordered_map<
						const cv::String*,
						const std::uniquePtr<const SliderConverter>>&
		slidersConverters();

	IControlPanelActions &performer;	///< the delegate responsible to perform selected actions
	const ISettings &cfg;				///< the settings, required to (re)initialize the sliders

	/// pointers to the names of the sliders that are undergoing value restoration
	std::unordered_set<const cv::String, std::hash<std::stringType>> slidersRestoringValue;

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
	int structuralSim, correlationCorrectness, underGlyphCorrectness, glyphEdgeCorrectness, asideGlyphCorrectness;
	int moreContrast, gravity, direction, largerSym;
	int thresh4Blanks;

	/**
	hack field

	The max of the encodings slider won't update unless
	issuing an additional slider move, which has to be ignored.
	*/
	bool updatingEncMax = false;

public:
	ControlPanel(IControlPanelActions &performer_, const ISettings &cfg_);
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
	void restoreSliderValue(const cv::String &trName, const std::stringType &errText) override;

	/// Authorizes the action of the control whose name is provided as parameter.
	/// When the action isn't allowed returns nullptr.
	std::uniquePtr<const ActionPermit> actionDemand(const cv::String &controlName) override;

	void updateEncodingsCount(unsigned uniqueEncodings) override;	///< puts also the slider on 0
	bool encMaxHack() const override final { return updatingEncMax; }		///< used for the hack above

	/// updates font size & encoding sliders, if necessary
	void updateSymSettings(unsigned encIdx, unsigned fontSz_) override;
	void updateImgSettings(const IfImgSettings &is) override; ///< updates sliders concerning IfImgSettings items
	void updateMatchSettings(const IMatchSettings &ms) override; ///< updates sliders concerning IMatchSettings items
};

#endif // H_CONTROL_PANEL

#endif // UNIT_TESTING
