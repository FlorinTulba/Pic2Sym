/**********************************************************
 Project:     Pic2Sym
 File:        controller.h

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_CONTROLLER
#define H_CONTROLLER

#ifdef UNIT_TESTING
class Controller final {
public:
	void reportGlyphProgress(double progress) {}
	void reportTransformationProgress(double progress) {}
};

#else // UNIT_TESTING not defined

#include "ui.h"
#include "transform.h"

/*
Manager of the views and data.
*/
class Controller final {
	// Data
	Transformer t;		// the model; KEEP IT BEFORE cp !!!
	Config &cfg;		// extracted from t (keep it after t)
	FontEngine &fe;		// extracted from t (keep it after t)
	Img &img;			// extracted from t (keep it after t)

	// Views
	Comparator comp;	// view for comparing original & result
	std::shared_ptr<CmapInspect> pCmi;	// view for inspecting the used cmap
	ControlPanel cp;	// the configuration view (keep it after t)

	// Validation flags
	bool imageOk = false, fontFamilyOk = false; // not set yet, so false
	bool hMaxSymsOk, vMaxSymsOk;
	bool fontSzOk;

	// Reports uncorrected settings when visualizing the cmap or while executing transform command.
	// Cmap visualization can ignore image-related errors by setting 'imageReguired' to false.
	bool validState(bool imageReguired = true) const;

	void updateCmapStatusBar() const;	// updates information about font family, encoding and size
	void symbolsChanged();				// triggered by new font family / encoding / size
	
	/*
	Shows a 'Please wait' window and reports the progress (0..1) as %.
	Details about the ongoing operation can be added to the title.
	*/
	void hourGlass(double progress, const std::string &title = "");

public:
	Controller(const std::string &cmd); // get the application path as parameter
	~Controller();						// destroys the windows

	// Waits for the user to press ESC and confirm he wants to leave
	void handleRequests() const;

	// Settings from view passed to model
	void newImage(const std::string &imgPath);
	void newFontFamily(const std::string &fontFile);
	void newHmaxSyms(int maxSyms);
	void newVmaxSyms(int maxSyms);
	void newFontEncoding(int encodingIdx);
	void newFontSize(int fontSz);
	void newThreshold4BlanksFactor(unsigned t);
	void newContrastFactor(double k);
	void newUnderGlyphCorrectnessFactor(double k);
	void newAsideGlyphCorrectnessFactor(double k);
	void newDirectionalSmoothnessFactor(double k);
	void newGravitationalSmoothnessFactor(double k);
	void newGlyphWeightFactor(double k);

	// Settings passed from model to view
	unsigned getHmaxSyms() const { return cfg.getMaxHSyms(); }
	unsigned getVmaxSyms() const { return cfg.getMaxVSyms(); }
	unsigned getFontSize() const { return cfg.getFontSz(); }
	unsigned getThreshold4BlanksFactor() const { return cfg.getBlankThreshold(); }
	double getContrastFactor() const { return cfg.get_kContrast(); }
	double getUnderGlyphCorrectnessFactor() const { return cfg.get_kSdevFg(); }
	double getAsideGlyphCorrectnessFactor() const { return cfg.get_kSdevBg(); }
	double getDirectionalSmoothnessFactor() const { return cfg.get_kCosAngleMCs(); }
	double getGravitationalSmoothnessFactor() const { return cfg.get_kMCsOffset(); }
	double getGlyphWeightFactor() const { return cfg.get_kGlyphWeight(); }
	Transformer::VVMatCItPair getFontFaces(unsigned from, unsigned maxCount) const;

	// Progress about loading, adapting glyphs
	void reportGlyphProgress(double progress);

	// Transformer
	void performTransformation();
	void reportTransformationProgress(double progress);
};
#endif // UNIT_TESTING not defined

#endif