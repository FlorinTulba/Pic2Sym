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

#ifndef H_CONTROLLER_BASE
#define H_CONTROLLER_BASE

#ifndef UNIT_TESTING
#include "pixMapSymBase.h"

#else // UNIT_TESTING defined
#include "std_memory.h"

#endif // UNIT_TESTING

#pragma warning ( push, 0 )

#include "std_string.h"

#pragma warning ( pop )

// Forward declarations
struct IUpdateSymSettings;
struct IGlyphsProgressTracker;
struct IPicTransformProgressTracker;
struct IPresentCmap;
struct IControlPanelActions;
struct IResizedImg;

/// Base interface for the Controller.
struct IController /*abstract*/ {
	virtual ~IController() = 0 {}

	// Group 1: 2 methods called so far only by ControlPanelActions
	virtual void symbolsChanged() = 0;	///< Triggered by new font family / encoding / size
	virtual void ensureExistenceCmapInspect() = 0;

	// Group 2: 2 methods called so far only by FontEngine
	virtual const IUpdateSymSettings& getUpdateSymSettings() const = 0;
	virtual const std::uniquePtr<const IPresentCmap>& getPresentCmap() const = 0; // the ref to uniquePtr solves a circular dependency inside the constructor

	// Last group of methods used by many different clients without an obvious pattern
	virtual const IGlyphsProgressTracker& getGlyphsProgressTracker() const = 0;
	virtual IPicTransformProgressTracker& getPicTransformProgressTracker() = 0;
	virtual const unsigned& getFontSize() const = 0; ///< font size determines grid size
	/// Returns true if transforming a new image or the last one, but under other image parameters
	virtual bool updateResizedImg(const IResizedImg &resizedImg_) = 0;
	/**
	Shows a 'Please wait' window and reports progress.

	@param progress the progress (0..1) as %
	@param title details about the ongoing operation
	@param async allows showing the window asynchronously
	*/
	virtual void hourGlass(double progress, const std::stringType &title = "", bool async = false) const = 0;
	virtual void showResultedImage(double completionDurationS) = 0; ///< Displays the resulted image

	/**
	Updates the status bar from the charmap inspector window.

	@param upperSymsCount an overestimated number of symbols from the unfiltered set
		or 0 when considering the exact number of symbols from the filtered set
	@param suffix an optional status bar message suffix
	@param async allows showing the new status bar message asynchronously
	*/
	virtual void updateStatusBarCmapInspect(unsigned upperSymsCount = 0U,
											const std::stringType &suffix = "",
											bool async = false) const = 0;

	/// Reports the duration of loading symbols / transforming images
	virtual void reportDuration(const std::stringType &text, double durationS) const = 0;

	virtual IControlPanelActions& getControlPanelActions() = 0;

#ifndef UNIT_TESTING
	/// Attempts to display 1st cmap page, when full. Called after appending each symbol from charmap. 
	virtual void display1stPageIfFull(const VPixMapSym &syms) = 0;
#endif // UNIT_TESTING not defined
};

#endif // H_CONTROLLER_BASE
