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

#ifndef H_CONTROLLER
#define H_CONTROLLER

#include "controllerBase.h"
#include "cmapPerspective.h"
#include "updateSymsActions.h"

#pragma warning ( push, 0 )

#include <atomic>

#pragma warning ( pop )

// Forward declarations
class FontEngine;
class MatchEngine;
struct ISettings;
struct ISettingsRW;
class SymSettings;
class Img;
class ControlPanel;
class AbsJobMonitor;
class PreselManager;
class Transformer;
class Comparator;
class CmapInspect;

/// Manager of the views and data.
class Controller : public IController {
protected:
	/// Responsible of updating symbol settings
	std::shared_ptr<const IUpdateSymSettings> updateSymSettings;
	
	/// Responsible for keeping track of the symbols loading process
	std::shared_ptr<const IGlyphsProgressTracker> glyphsProgressTracker;

	/// Responsible for monitoring the progress during an image transformation
	std::shared_ptr<IPicTransformProgressTracker> picTransformProgressTracker;


	// Control of displayed progress
	std::shared_ptr<AbsJobMonitor> glyphsUpdateMonitor;	///< in charge of displaying the progress while updating the glyphs
	std::shared_ptr<AbsJobMonitor> imgTransformMonitor;	///< in charge of displaying the progress while transforming images

	CmapPerspective cmP;	///< reorganized symbols to be visualized within the cmap viewer

	/**
	Provides read-only access to Cmap data.
	Needed by 'fe' from below.
	Uses 'me' from below and 'cmP' from above
	*/
	std::shared_ptr<const IPresentCmap> presentCmap;

	// Data
	/// pointer to the resized version of most recent image that had to be transformed
	std::shared_ptr<const ResizedImg> resizedImg;
	FontEngine &fe;		///< font engine
	ISettingsRW &cfg;	///< the settings for the transformations
	MatchEngine &me;	///< matching engine
	Transformer &t;		///< transforming engine
	PreselManager &pm;	///< preselection manager

	// Views
	Comparator &comp;					///< view for comparing original & result
	std::shared_ptr<CmapInspect> pCmi;	///< view for inspecting the used cmap

	/// Allows saving a selection of symbols pointed within the charmap viewer
	std::shared_ptr<const ISelectSymbols> selectSymbols;

	/// Responsible for the actions triggered by the controls from Control Panel
	std::shared_ptr<IControlPanelActions> controlPanelActions;

	// synchronization items necessary while updating symbols
	mutable LockFreeQueue updateSymsActionsQueue;
	std::atomic_flag updatingSymbols;	///< stays true while updating the symbols
	std::atomic_flag updating1stCmapPage;	///< controls concurrent attempts to update 1st page
	/// Stores the events occurred while updating the symbols.
	/// queue requires template param with trivial destructor and assign operator,
	/// so shared_ptr isn't useful here

	const std::string textForCmapStatusBar(unsigned upperSymsCount = 0U) const; ///< status bar with font information
	const std::string textHourGlass(const std::string &prefix, double progress) const; ///< progress

#ifdef UNIT_TESTING
public: // Providing get<field> as public for Unit Testing
#endif // UNIT_TESTING defined
	// Methods for initialization
	static Comparator& getComparator();
	FontEngine& getFontEngine(const SymSettings &ss_) const;
	MatchEngine& getMatchEngine(const ISettings &cfg_);
	Transformer& getTransformer(const ISettings &cfg_);
	PreselManager& getPreselManager(const ISettings &cfg_);

public:
	Controller(ISettingsRW &s);	///< Initializes controller with ISettingsRW object s
	Controller(const Controller&) = delete;
	void operator=(const Controller&) = delete;
	~Controller();				///< destroys the windows

	std::shared_ptr<const IUpdateSymSettings> getUpdateSymSettings() const override;
	std::shared_ptr<const IGlyphsProgressTracker> getGlyphsProgressTracker() const override;
	std::shared_ptr<IPicTransformProgressTracker> getPicTransformProgressTracker() override;
	std::shared_ptr<const IPresentCmap> getPresentCmap() const override;
	std::shared_ptr<const ISelectSymbols> getSelectSymbols() const override;
	std::shared_ptr<IControlPanelActions> getControlPanelActions() override;

	/// Waits for the user to press ESC and confirm he wants to leave
	static void handleRequests();

	const unsigned& getFontSize() const override; ///< font size determines grid size

	void symbolsChanged() override;	///< triggered by new font family / encoding / size

	/// Returns true if transforming a new image or the last one, but under other image parameters
	bool updateResizedImg(std::shared_ptr<const ResizedImg> resizedImg_) override;

	/**
	Shows a 'Please wait' window and reports progress.

	@param progress the progress (0..1) as %
	@param title details about the ongoing operation
	@param async allows showing the window asynchronously
	*/
	void hourGlass(double progress, const std::string &title = "", bool async = false) const override;

	/**
	Updates the status bar from the charmap inspector window.

	@param upperSymsCount an overestimated number of symbols from the unfiltered set
	or 0 when considering the exact number of symbols from the filtered set
	@param suffix an optional status bar message suffix
	@param async allows showing the new status bar message asynchronously
	*/
	void updateStatusBarCmapInspect(unsigned upperSymsCount = 0U,
									const std::string &suffix = "",
									bool async = false) const override;
	/// Reports the duration of loading symbols / transforming images
	void reportDuration(const std::string &text, double durationS) const override;

	/// Attempts to display 1st cmap page, when full. Called after appending each symbol from charmap. 
	void display1stPageIfFull(const VPixMapSym &syms) override;

	void showResultedImage(double completionDurationS) override; ///< Displays the resulted image
};

#endif // H_CONTROLLER
