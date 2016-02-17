/**********************************************************
 Project:     Pic2Sym
 File:        controller.h

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_CONTROLLER
#define H_CONTROLLER

#include "ui.h"
#include "transform.h"

#include <chrono>

class Controller;

// Envelopes all parameters required for transforming images
class Settings {
	SymSettings ss;		// parameters concerning the symbols set used for approximating patches
	ImgSettings is;		// contains max count of horizontal & vertical patches to process
	MatchSettings ms;	// settings used during approximation process
	friend class Controller; // the unique setter of the fields above (apart serialization)

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

	/*
	Creates a complete set of settings required during image transformations.
	The incoming parameter ms_ is completely moved to ms field, so that Settings to be
	the only setter of the MatchSettings in use.
	*/
	Settings(const MatchSettings &&ms_);

	const SymSettings& symSettings() const { return ss; }
	const ImgSettings& imgSettings() const { return is; }
	const MatchSettings& matchSettings() const { return ms; }

	friend std::ostream& operator<<(std::ostream &os, const Settings &s);
};

BOOST_CLASS_VERSION(Settings, 0)

// Manager of the views and data.
class Controller final {
	// Data
	Img &img;			// image to process
	FontEngine &fe;		// font engine
	Settings &cfg;		// the settings for the transformations
	MatchEngine &me;	// matching engine
	Transformer &t;		// transforming engine

	// Views
	Comparator &comp;					// view for comparing original & result
	std::shared_ptr<CmapInspect> pCmi;	// view for inspecting the used cmap
	ControlPanel &cp;					// the configuration view

	// Validation flags
	bool imageOk = false, fontFamilyOk = false; // not set yet, so false
	bool hMaxSymsOk, vMaxSymsOk;
	bool fontSzOk;

	// Reports uncorrected settings when visualizing the cmap or while executing transform command.
	// Cmap visualization can ignore image-related errors by setting 'imageReguired' to false.
	bool validState(bool imageRequired = true) const;

	void updateCmapStatusBar() const;	// updates information about font family, encoding and size
	void symbolsChanged();				// triggered by new font family / encoding / size

	// called by FontEngine::newFont after installing a new font to update SymSettings
	void selectedFontFile(const std::string &fName) const;
	friend bool FontEngine::newFont(const std::string&);

	// called by FontEngine::setNthUniqueEncoding to update the encoding in SymSettings
	void selectedEncoding(const std::string &encName) const;
	friend bool FontEngine::setNthUniqueEncoding(unsigned);

	/*
	Shows a 'Please wait' window and reports the progress (0..1) as %.
	Details about the ongoing operation can be added to the title.
	*/
	void hourGlass(double progress, const std::string &title = "") const;

	// called by friend class Timer from below when starting and ending the computations
	void symsSetUpdate(bool done = false, double elapsed = 0.) const;	// updating symbols set
	void imgTransform(bool done = false, double elapsed = 0.) const;	// transforming an image

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
	Controller(Settings &s);	// get the application path as parameter
	~Controller();				// destroys the windows

	void restoreUserDefaultMatchSettings();
	void setUserDefaultMatchSettings() const;

	void loadSettings();
	void saveSettings() const;

	// Waits for the user to press ESC and confirm he wants to leave
	static void handleRequests();

	// Settings from view passed to model
	void newImage(const std::string &imgPath);
	void newFontFamily(const std::string &fontFile);
	void newFontEncoding(int encodingIdx);
	bool newFontEncoding(const std::string &encName);
	void newFontSize(int fontSz);
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

	// Settings passed from model to view
	unsigned getFontSize() const { return cfg.ss.getFontSz(); }
	MatchEngine::VSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const;

	// Progress about loading, adapting glyphs
	void reportGlyphProgress(double progress) const;

	// Transformer
	bool performTransformation();
	void reportTransformationProgress(double progress) const;

	// Timing cmap updates and also image transformations
	class Timer {
	public:
		enum struct ComputationType : unsigned char {
			SYM_SET_UPDATE,	// used when updating the symbols set
			IMG_TRANSFORM	// used when transforming an image
		};
	private:
		const Controller &ctrler;
		const ComputationType compType;

		// the moment when computation started
		const std::chrono::time_point<std::chrono::high_resolution_clock> start;

		bool active = true;		// true as long as not released

	public:
		Timer(const Controller &ctrler_, ComputationType compType_);	// initializes start
		~Timer();				// if not released, reports duration

		void release();			// stops the timer
	};

	friend class Timer;

#ifdef UNIT_TESTING
	// Methods available only in Unit Testing mode
	bool newImage(const cv::Mat &imgMat);	// Provide directly a matrix instead of an image
#endif
};

#endif