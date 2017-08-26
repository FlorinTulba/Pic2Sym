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
struct IFontEngine;
struct IMatchEngine;
struct ISettings;
struct ISettingsRW;
struct ISymSettings;
class Img;
class AbsJobMonitor;
struct ITransformCompletion;
struct ITransformer;
struct IComparator;
struct ICmapInspect;
struct ISelectSymbols;

/**
Base interface for the Controller.

There appear now 3 segregated groups of classes (as reported by AI Reviewer).

The realization of most of these methods involves many overlapping fields
and in the future some of the methods might:
- get split / merged
- be used by unexpected / different clients

Furthermore, 2 of the observed groups of methods contain only 2 methods each, which is negligible.


To still address the segregation, here would be some options:

A.	For a fast implementation, good adaptability to unforeseen factors
	and with a small cost of creating a `Concentrator class`:

		IController could inherit 2 new parallel interfaces,
		say IControllerSupportForCPA and IControllerSupportForFE
		of the 2 smallest groups of methods (for ControlPanelActions and for FontEngine).
		
	These interfaces can be split / merged / extended or changed rather easily.

B.	A more complex implementation and rather inflexible to unforeseen factors
	but without the `Concentrator class` issue:

		Separating IController into inheritance layers:

		- IControllerCore - to contain the 3rd (largest) group of methods.
		  Its implementation would encapsulate as protected most fields (which are strictly required)
		  from the existing Controller class

		- IControllerLayer1 - to contain the methods from say group 1.
		  Its implementation would be on top of the core class and might not need any new fields

		- IControllerLayer2 - to contain the methods from the remaining group.
		  Its implementation would be on top of the previous layer and
		  will encapsulate any remaining required field

	This `vertical` approach might later involve costly operations:
	- moving fields around between layers, plus splitting methods between layers
	- switching layers

C.	Aggregation of 2 providers for the interfaces for the 2 smallest groups of methods.
	The interface provider of group 2 needs only 2 fields from Controller.
	The interface provider of group 1 needs either to be a friend of Controller or
	to receive all the required fields in the constructor.

	This approach is nothing more than `Feature Envy` and brings high maintenance costs.
*/
class Controller : public IController {
protected:
	/// Responsible of updating symbol settings
	std::sharedPtr<const IUpdateSymSettings> updateSymSettings;
	
	/// Responsible for keeping track of the symbols loading process
	std::sharedPtr<const IGlyphsProgressTracker> glyphsProgressTracker;

	/// Responsible for monitoring the progress during an image transformation
	std::sharedPtr<IPicTransformProgressTracker> picTransformProgressTracker;


	// Control of displayed progress
	std::sharedPtr<AbsJobMonitor> glyphsUpdateMonitor;	///< in charge of displaying the progress while updating the glyphs
	std::sharedPtr<AbsJobMonitor> imgTransformMonitor;	///< in charge of displaying the progress while transforming images

	CmapPerspective cmP;	///< reorganized symbols to be visualized within the cmap viewer

	/**
	Provides read-only access to Cmap data.
	Needed by 'fe' from below.
	Uses 'me' from below and 'cmP' from above
	*/
	std::sharedPtr<const IPresentCmap> presentCmap;

	// Data
	/// pointer to the resized version of most recent image that had to be transformed
	std::sharedPtr<const IResizedImg> resizedImg;
	IFontEngine &fe;		///< font engine
	ISettingsRW &cfg;		///< the settings for the transformations
	IMatchEngine &me;		///< matching engine
	ITransformCompletion &t;///< results of the transformation

	// Views
	IComparator &comp;					///< view for comparing original & result
	std::sharedPtr<ICmapInspect> pCmi;	///< view for inspecting the used cmap

	/// Allows saving a selection of symbols pointed within the charmap viewer
	std::sharedPtr<const ISelectSymbols> selectSymbols;

	/// Responsible for the actions triggered by the controls from Control Panel
	std::sharedPtr<IControlPanelActions> controlPanelActions;

	// synchronization items necessary while updating symbols
	mutable LockFreeQueue updateSymsActionsQueue;
	std::atomic_flag updatingSymbols;	///< stays true while updating the symbols
	std::atomic_flag updating1stCmapPage;	///< controls concurrent attempts to update 1st page
	/// Stores the events occurred while updating the symbols.
	/// queue requires template param with trivial destructor and assign operator,
	/// so sharedPtr isn't useful here

	const std::stringType textForCmapStatusBar(unsigned upperSymsCount = 0U) const; ///< status bar with font information
	const std::stringType textHourGlass(const std::stringType &prefix, double progress) const; ///< progress

#ifdef UNIT_TESTING
public: // Providing get<field> as public for Unit Testing
#endif // UNIT_TESTING defined
	// Methods for initialization
	static IComparator& getComparator();
	IFontEngine& getFontEngine(const ISymSettings &ss_) const;
	IMatchEngine& getMatchEngine(const ISettings &cfg_);
	ITransformer& getTransformer(const ISettings &cfg_);

public:
	Controller(ISettingsRW &s);	///< Initializes controller with ISettingsRW object s
	Controller(const Controller&) = delete;
	void operator=(const Controller&) = delete;
	~Controller();				///< destroys the windows

	/// Waits for the user to press ESC and confirm he wants to leave
	static void handleRequests();

	// Group 1: 2 methods called so far only by ControlPanelActions
	void ensureExistenceCmapInspect() override;
	void symbolsChanged() override;	///< triggered by new font family / encoding / size

	// Group 2: 2 methods called so far only by FontEngine
	std::sharedPtr<const IUpdateSymSettings> getUpdateSymSettings() const override;
	const std::sharedPtr<const IPresentCmap>& getPresentCmap() const override;

	// Last group of methods used by many different clients without an obvious pattern
	std::sharedPtr<const IGlyphsProgressTracker> getGlyphsProgressTracker() const override;
	std::sharedPtr<IPicTransformProgressTracker> getPicTransformProgressTracker() override;
	std::sharedPtr<IControlPanelActions> getControlPanelActions() override;

	const unsigned& getFontSize() const override; ///< font size determines grid size

	/// Returns true if transforming a new image or the last one, but under other image parameters
	bool updateResizedImg(std::sharedPtr<const IResizedImg> resizedImg_) override;

	/**
	Shows a 'Please wait' window and reports progress.

	@param progress the progress (0..1) as %
	@param title details about the ongoing operation
	@param async allows showing the window asynchronously
	*/
	void hourGlass(double progress, const std::stringType &title = "", bool async = false) const override;

	/**
	Updates the status bar from the charmap inspector window.

	@param upperSymsCount an overestimated number of symbols from the unfiltered set
	or 0 when considering the exact number of symbols from the filtered set
	@param suffix an optional status bar message suffix
	@param async allows showing the new status bar message asynchronously
	*/
	void updateStatusBarCmapInspect(unsigned upperSymsCount = 0U,
									const std::stringType &suffix = "",
									bool async = false) const override;
	/// Reports the duration of loading symbols / transforming images
	void reportDuration(const std::stringType &text, double durationS) const override;

	void showResultedImage(double completionDurationS) override; ///< Displays the resulted image

#ifndef UNIT_TESTING
	/// Attempts to display 1st cmap page, when full. Called after appending each symbol from charmap. 
	void display1stPageIfFull(const VPixMapSym &syms) override;
#endif // UNIT_TESTING not defined
};

#endif // H_CONTROLLER
