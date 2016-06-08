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

#ifndef H_CONTROLLER
#define H_CONTROLLER

#include "imgSettings.h"
#include "views.h"
#include "transform.h"
#include "controlPanelActions.h"
#include "presentCmap.h"
#include "validateFont.h"
#include "glyphsProgressTracker.h"
#include "picTransformProgressTracker.h"
#include "updateSymsActions.h"
#include "misc.h"
#include "timing.h"

#include <atomic>

class ControlPanel; // forward declaration

/// Manager of the views and data.
class Controller :
	public IControlPanelActions, public IPresentCmap, public IValidateFont,
	public IGlyphsProgressTracker, public IPicTransformProgressTracker {
protected:
	// Data
	Img &img;			///< original image to process after resizing

	/// pointer to the resized version of most recent image that had to be transformed
	std::shared_ptr<const ResizedImg> resizedImg;
	FontEngine &fe;		///< font engine
	Settings &cfg;		///< the settings for the transformations
	MatchEngine &me;	///< matching engine
	Transformer &t;		///< transforming engine

	// Views
	Comparator &comp;					///< view for comparing original & result
	std::shared_ptr<CmapInspect> pCmi;	///< view for inspecting the used cmap
	ControlPanel &cp;					///< the configuration view

	// Validation flags
	bool imageOk = false, fontFamilyOk = false; // not set yet, so false
	bool hMaxSymsOk, vMaxSymsOk;
	bool fontSzOk;

	// synchronization items necessary while updating symbols
	mutable LockFreeQueueSz22 updateSymsActionsQueue;
	std::atomic_flag updatingSymbols;	///< stays true while updating the symbols
	std::atomic_flag updating1stCmapPage;	///< controls concurrent attempts to update 1st page
	/// Stores the events occurred while updating the symbols.
	/// queue requires template param with trivial destructor and assign operator,
	/// so shared_ptr isn't useful here

	/// Reports uncorrected settings when visualizing the cmap or while executing transform command.
	/// Cmap visualization can ignore image-related errors by setting 'imageReguired' to false.
	bool validState(bool imageRequired = true) const;

	const std::string textForCmapStatusBar(unsigned upperSymsCount = 0U) const; ///< status bar with font information
	const std::string textForCmapOverlay(double elapsed) const; ///< Glyph loading duration
	const std::string textForComparatorOverlay(double elapsed) const; ///< Transformation duration
	const std::string textHourGlass(const std::string &prefix, double progress) const; ///< progress

	void symbolsChanged();				///< triggered by new font family / encoding / size

	/**
	Shows a 'Please wait' window and reports progress.

	@param progress the progress (0..1) as %
	@param title details about the ongoing operation
	*/
	void hourGlass(double progress, const std::string &title = "") const;

	// Next 3 private methods do the ground work for their public correspondent methods
	bool _newFontFamily(const std::string &fontFile, bool forceUpdate = false);
	bool _newFontEncoding(const std::string &encName, bool forceUpdate = false);
	bool _newFontSize(int fontSz, bool forceUpdate = false);

#ifdef UNIT_TESTING
public: // Providing get<field> as public for Unit Testing
#endif
	// Methods for initialization
	static Img& getImg();
	static Comparator& getComparator();
	FontEngine& getFontEngine(const SymSettings &ss_) const;
	MatchEngine& getMatchEngine(const Settings &cfg_) const;
	Transformer& getTransformer(const Settings &cfg_) const;
	ControlPanel& getControlPanel(Settings &cfg_);

public:
	Controller(Settings &s);	///< Initializes controller with Settings object s
	~Controller();				///< destroys the windows

	/// Waits for the user to press ESC and confirm he wants to leave
	static void handleRequests();

	// IControlPanelActions implementation below
	/// overwriting MatchSettings with the content of 'initMatchSettings.cfg'
	void restoreUserDefaultMatchSettings() override;
	void setUserDefaultMatchSettings() const override; ///< saving current MatchSettings to 'initMatchSettings.cfg'
	void loadSettings() override;		///< updating the Settings object
	void saveSettings() const override;	///< saving the Settings object
	void newImage(const std::string &imgPath) override;
	void newFontFamily(const std::string &fontFile) override;
	void newFontEncoding(int encodingIdx) override;
	bool newFontEncoding(const std::string &encName) override;
	void newFontSize(int fontSz) override;
	void newSymsBatchSize(int symsBatchSz) override;
	void newStructuralSimilarityFactor(double k) override;
	void newUnderGlyphCorrectnessFactor(double k) override;
	void newGlyphEdgeCorrectnessFactor(double k) override;
	void newAsideGlyphCorrectnessFactor(double k) override;
	void newContrastFactor(double k) override;
	void newGravitationalSmoothnessFactor(double k) override;
	void newDirectionalSmoothnessFactor(double k) override;
	void newGlyphWeightFactor(double k) override;
	void newThreshold4BlanksFactor(unsigned t) override;
	void newHmaxSyms(int maxSyms) override;
	void newVmaxSyms(int maxSyms) override;
	/**
	Sets the result mode:
	- approximations only (actual result) - patches become symbols, with no cosmeticizing.
	- hybrid (cosmeticized result) - for displaying approximations blended with a blurred version of the original. The better an approximation, the fainter the hint background

	@param hybrid boolean: when true, establishes the cosmeticized mode; otherwise leaves the actual result as it is
	*/
	void setResultMode(bool hybrid) override;
	bool performTransformation() override;
	void showAboutDlg(const std::string &title, const std::wstring &content) override;
	void showInstructionsDlg(const std::string &title, const std::wstring &content) override;

	// Implementation of IPresentCmap below
	void resetCmapView() override;
	void display1stPageIfFull(const std::vector<const PixMapSym> &syms) override;
	unsigned getFontSize() const override;
	MatchEngine::VSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const override;
	const std::set<unsigned>& getClusterOffsets() const override;
	void showUnofficialSymDetails(unsigned symsCount) const override;

	// Implementation of IValidateFont below
	/// called by FontEngine::newFont after installing a new font to update SymSettings
	void selectedFontFile(const std::string &fName) const override;
	/// called by FontEngine::setNthUniqueEncoding to update the encoding in SymSettings
	void selectedEncoding(const std::string &encName) const override;

	// Implementation of IGlyphsProgressTracker below
	/// Report progress about loading, adapting glyphs
	void reportGlyphProgress(double progress) const override;
	void updateSymsDone(double durationS) const override;
	Timer createTimerForGlyphs() const override; ///< Creates the monitor to time the glyph loading and preprocessing
	void reportSymsUpdateDuration(double elapsed) const override;

	// Implementation of IPicTransformProgressTracker below
	bool updateResizedImg(std::shared_ptr<const ResizedImg> resizedImg_) override;
	void reportTransformationProgress(double progress, bool showDraft = false) const override;
	void presentTransformationResults(double completionDurationS = -1.) const override;
	Timer createTimerForImgTransform() const override; ///< Creates the monitor to time the picture approximation process

	/// Base class for TimerActions_SymSetUpdate and TimerActions_ImgTransform
	struct TimerActions_Controller : ITimerActions {
	protected:
		const Controller &ctrler; ///< actual manager of the events

		TimerActions_Controller(const Controller &ctrler_);
	};

	/// Actions for start & stop chronometer while timing glyphs loading & preprocessing
	struct TimerActions_SymSetUpdate : TimerActions_Controller {
		TimerActions_SymSetUpdate(const Controller &ctrler_);

		void onStart() override;	///< action to be performed when the timer is started

		/// action to be performed when the timer is released/deleted
		/// @param elapsedS total elapsed time in seconds
		void onRelease(double elapsedS) override;
	};

	/// Actions for start & stop chronometer while timing the approximation of the picture
	struct TimerActions_ImgTransform : TimerActions_Controller {
		TimerActions_ImgTransform(const Controller &ctrler_);

		void onStart() override;	///< action to be performed when the timer is started

		/// action to be performed when the timer is released/deleted
		/// @param elapsedS total elapsed time in seconds
		void onRelease(double elapsedS) override;

		/// action to be performed when the timer is canceled
		/// @param reason explanation for cancellation
		void onCancel(const std::string &reason = "") override;
	};

#ifdef UNIT_TESTING
	// Method available only in Unit Testing mode
	bool newImage(const cv::Mat &imgMat);	///< Provide directly a matrix instead of an image
#endif
};

#endif