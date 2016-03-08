/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-22
 and belongs to the Pic2Sym project.

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

#include "ui.h"
#include "transform.h"

#include <chrono>

class Controller;

/// Envelopes all parameters required for transforming images
class Settings {
	SymSettings ss;		///< parameters concerning the symbols set used for approximating patches
	ImgSettings is;		///< contains max count of horizontal & vertical patches to process
	MatchSettings ms;	///< settings used during approximation process
	friend class Controller; ///< the unique setter of ss, is, ms (apart serialization)

	/**
	Loads or saves a Settings object.

	@param ar the source/target archive
	@param version When loading (overwriting *this with the Settings from ar),
	it represents the version of the object loaded from ar;
	When saving to ar, it's the last version of Settings
	*/
	template<class Archive>
	void serialize(Archive &ar, const unsigned version) {
		ar & ss & is & ms;
	}
	friend class boost::serialization::access;

public:
	static const unsigned // Limits  
		MIN_FONT_SIZE = 7U, MAX_FONT_SIZE = 50U, DEF_FONT_SIZE = 10U,
		MAX_THRESHOLD_FOR_BLANKS = 50U,
		MIN_H_SYMS = 3U, MAX_H_SYMS = 1024U,
		MIN_V_SYMS = 3U, MAX_V_SYMS = 768U;

	static bool isBlanksThresholdOk(unsigned t) { return t < MAX_THRESHOLD_FOR_BLANKS; }
	static bool isHmaxSymsOk(unsigned syms) { return syms>=MIN_H_SYMS && syms<=MAX_H_SYMS; }
	static bool isVmaxSymsOk(unsigned syms) { return syms>=MIN_V_SYMS && syms<=MAX_V_SYMS; }
	static bool isFontSizeOk(unsigned fs) { return fs>=MIN_FONT_SIZE && fs<=MAX_FONT_SIZE; }

	/**
	Creates a complete set of settings required during image transformations.

	@param ms_ incoming parameter completely moved to ms field, so that Settings to be
	the only setter of the MatchSettings in use.
	*/
	Settings(const MatchSettings &&ms_);

	const SymSettings& symSettings() const { return ss; }
	const ImgSettings& imgSettings() const { return is; }
	const MatchSettings& matchSettings() const { return ms; }

	friend std::ostream& operator<<(std::ostream &os, const Settings &s);
};

BOOST_CLASS_VERSION(Settings, 0)

/// Manager of the views and data.
class Controller final {
	// Data
	Img &img;			///< image to process
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

	/// Reports uncorrected settings when visualizing the cmap or while executing transform command.
	/// Cmap visualization can ignore image-related errors by setting 'imageReguired' to false.
	bool validState(bool imageRequired = true) const;

	void updateCmapStatusBar() const;	///< updates information about font family, encoding and size
	void symbolsChanged();				///< triggered by new font family / encoding / size

	/// called by FontEngine::newFont after installing a new font to update SymSettings
	void selectedFontFile(const std::string &fName) const;
	friend bool FontEngine::newFont(const std::string&);

	/// called by FontEngine::setNthUniqueEncoding to update the encoding in SymSettings
	void selectedEncoding(const std::string &encName) const;
	friend bool FontEngine::setNthUniqueEncoding(unsigned);

	/**
	Shows a 'Please wait' window and reports progress.

	@param progress the progress (0..1) as %
	@param title details about the ongoing operation
	*/
	void hourGlass(double progress, const std::string &title = "") const;

	/// Called by friend class Timer from below when starting and ending the update of the symbol set
	void symsSetUpdate(bool done = false, double elapsed = 0.) const;

	/// Called by friend class Timer from below when starting and ending the image transformation
	void imgTransform(bool done = false, double elapsed = 0.) const;

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

	/// overwriting MatchSettings with the content of 'initMatchSettings.cfg'
	void restoreUserDefaultMatchSettings();		
	void setUserDefaultMatchSettings() const; ///< saving current MatchSettings to 'initMatchSettings.cfg'

	void loadSettings();		///< updating the Settings object
	void saveSettings() const;	///< saving the Settings object

	/// Waits for the user to press ESC and confirm he wants to leave
	static void handleRequests();

	// Settings from view passed to model
	void newImage(const std::string &imgPath);
	void newFontFamily(const std::string &fontFile);
	void newFontEncoding(int encodingIdx);
	bool newFontEncoding(const std::string &encName);
	void newFontSize(int fontSz);
	void newStructuralSimilarityFactor(double k);
	void newUnderGlyphCorrectnessFactor(double k);
	void newGlyphEdgeCorrectnessFactor(double k);
	void newAsideGlyphCorrectnessFactor(double k);
	void newContrastFactor(double k);
	void newGravitationalSmoothnessFactor(double k);
	void newDirectionalSmoothnessFactor(double k);
	void newGlyphWeightFactor(double k);
	void newThreshold4BlanksFactor(unsigned t);
	void newHmaxSyms(int maxSyms);
	void newVmaxSyms(int maxSyms);
	/**
	Sets the result mode:
	- approximations only (actual result) - patches become symbols, with no cosmeticizing.
	- hybrid (cosmeticized result) - for displaying approximations blended with a blurred version of the original. The better an approximation, the fainter the hint background

	@param hybrid boolean: when true, establishes the cosmeticized mode; otherwise leaves the actual result as it is
	*/
	void setResultMode(bool hybrid);

	// Settings passed from model to view
	unsigned getFontSize() const { return cfg.ss.getFontSz(); }
	MatchEngine::VSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const;

	/// Report progress about loading, adapting glyphs
	void reportGlyphProgress(double progress) const;

	// Transformer
	bool performTransformation();
	void reportTransformationProgress(double progress) const;

	/// Timing cmap updates and also image transformations
	class Timer {
	public:
		enum struct ComputationType : unsigned char {
			SYM_SET_UPDATE,	// used when updating the symbols set
			IMG_TRANSFORM	// used when transforming an image
		};
	private:
		const Controller &ctrler;
		const ComputationType compType;

		/// the moment when computation started
		const std::chrono::time_point<std::chrono::high_resolution_clock> start;

		bool active = true;		///< true as long as not released

	public:
		Timer(const Controller &ctrler_, ComputationType compType_);	///< initializes start
		~Timer();				///< if not released, reports duration

		void release();			///< stops the timer
	};

	friend class Timer;

#ifdef UNIT_TESTING
	// Method available only in Unit Testing mode
	bool newImage(const cv::Mat &imgMat);	///< Provide directly a matrix instead of an image
#endif
};

#endif