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

// Manager of the views and data.
class Controller final {
	// Data
	Img &img;			// image to process
	FontEngine &fe;		// font engine
	Config &cfg;		// most settings for the transformations
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
	bool validState(bool imageReguired = true) const;

	void updateCmapStatusBar() const;	// updates information about font family, encoding and size
	void symbolsChanged();				// triggered by new font family / encoding / size
	
	/*
	Shows a 'Please wait' window and reports the progress (0..1) as %.
	Details about the ongoing operation can be added to the title.
	*/
	void hourGlass(double progress, const std::string &title = "") const;

#ifdef UNIT_TESTING
public: // Providing get<field> as public for Unit Testing
#endif
	// Methods for initialization
	static Img& getImg();
	FontEngine& getFontEngine() const;
	MatchEngine& getMatchEngine(const Config &cfg_) const;
	Transformer& getTransformer(const Config &cfg_) const;
	Comparator& getComparator() const;
	ControlPanel& getControlPanel(Config &cfg_);

public:
	Controller(Config &cfg_);	// get the application path as parameter
	~Controller();				// destroys the windows

	// Waits for the user to press ESC and confirm he wants to leave
	void handleRequests() const;

	// Settings from view passed to model
	void newImage(const std::string &imgPath);
	void newFontFamily(const std::string &fontFile);
	void newFontEncoding(int encodingIdx);
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
	unsigned getFontSize() const { return cfg.getFontSz(); }
	MatchEngine::VSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const;

	// Progress about loading, adapting glyphs
	void reportGlyphProgress(double progress) const;

	// Transformer
	bool performTransformation();
	void reportTransformationProgress(double progress) const;

#ifdef UNIT_TESTING
	// Methods available only in Unit Testing mode
	bool newImage(const cv::Mat &imgMat);	// Provide directly a matrix instead of an image
	bool newFontEncoding(const std::string &encName); // Use an Encoding name
#endif
};

#endif