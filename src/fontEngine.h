/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#ifndef H_FONT_ENGINE
#define H_FONT_ENGINE

#include "fontEngineBase.h"

#pragma warning(push, 0)

#include <unordered_set>

#include <filesystem>

#include <boost/bimap/bimap.hpp>

#pragma warning(pop)

extern template class std::unordered_set<FT_ULong>;

// Forward declarations
class IController;
class IUpdateSymSettings;
class IPresentCmap;
class ISymSettings;
class IPmsCont;

/// FontEngine class wraps some necessary FreeType functionality.
class FontEngine : public IFontEngine {
 public:
  /**
  Constructs a FontEngine

  @param ctrler_ font validation and glyph preprocessing monitor aspects of the
  Controller
  @param ss_ font data

  @throw runtime_error if FreeType cannot initialize

  Exception to be only reported, not handled
  */
  FontEngine(IController& ctrler_, const ISymSettings& ss_) noexcept;
  ~FontEngine() noexcept;

  FontEngine(const FontEngine&) = delete;
  FontEngine(FontEngine&&) = delete;
  void operator=(const FontEngine&) = delete;
  void operator=(FontEngine&&) = delete;

  /// When unable to process a font type, invalidate it completely
  void invalidateFont() noexcept override;

  /// Tries to use the font from 'fontFile_'
  bool newFont(const std::string& fontFile_) noexcept override;

  /**
  Sets the desired font height in pixels
  @throw logic_error for an incomplete font configuration
  @throw invalid_argument for fontSz_ outside the range specified in the
  settings or the value cannot actually be used for the glyphs
  @runtime_error when there are issues loading the resized glyphs

  Exceptions to be only reported, not handled
  */
  void setFontSz(unsigned fontSz_) noexcept(!UT) override;

  /**
  Sets an encoding by name
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  bool setEncoding(const std::string& encName,
                   bool forceUpdate = false) noexcept(!UT) override;

  /**
  Switches to nth unique encoding
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  bool setNthUniqueEncoding(unsigned idx) noexcept(!UT) override;

  /**
  Upper bound of symbols count in the cmap
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  unsigned upperSymsCount() const noexcept(!UT) override;

  /**
  Get the symbols set
  @throw logic_error for an incompletely-loaded font set

  Exception to be only reported, not handled
  */
  const VPixMapSym& symsSet() const noexcept(!UT) override;

  /**
  Get coverageOfSmallGlyphs
  @throw logic_error for an incompletely-loaded font set

  Exception to be only reported, not handled
  */
  double smallGlyphsCoverage() const noexcept(!UT) override;

  /// Font name provided by Font Dialog
  const std::string& fontFileName() const noexcept override;

  /**
  @return the count of unique encodings
  @throw logic_error for incomplete font configuration

  Exception to be only reported, not handled
  */
  unsigned uniqueEncodings() const noexcept(!UT) override;

  /**
  Gets current encoding

  @param pEncodingIndex if not nullptr, address where to store the index of the
  current encoding

  @return current encoding

  @throw logic_error for incomplete font configuration

  Exception to be only reported, not handled
  */
  const std::string& getEncoding(unsigned* pEncodingIndex = nullptr) const
      noexcept(!UT) override;

  FT_String* getFamily() const noexcept override;  ///< get font family
  FT_String* getStyle() const noexcept override;   ///< get font style

  /**
  @return the type of the symbols, independent of font size

  @throw logic_error only in UnitTesting for incomplete font configuration

  Exception to be only reported, not handled
  */
  std::string getFontType() const noexcept(!UT) override;

  /**
  (Creates/Loads and) Returns(/Saves) the small versions of all the symbols from
  current cmap

  @return a list of tiny symbols from current cmap

  @throw logic_error if called before setAsReady()
  @throw TinySymsLoadingFailure for problems loading/processing tiny symbols,
  which might be difficult to handle (they belong to the FreeType project)

  Exceptions to be only reported, not handled
  */
  const VTinySyms& getTinySyms() noexcept(!UT) override;

  void disposeTinySyms() noexcept override;  ///< Releases tiny symbols data

  /**
  Determines if fontType was already processed.
  The path to the file supposed to contain the desired tiny symbols data
  is returned in the tinySymsDataFile parameter.
  */
  static bool isTinySymsDataSavedOnDisk(
      const std::string& fontType,
      std::filesystem::path& tinySymsDataFile) noexcept;

  /// Setting the symbols monitor
  FontEngine& useSymsMonitor(AbsJobMonitor& symsMonitor_) noexcept override;

  PROTECTED :

      /**
      Validates a new font file.

      When fName is valid, face_ parameter will return the successfully loaded
      font.
      */
      bool
      checkFontFile(const std::filesystem::path& fontPath, FT_Face& face_) const
      noexcept;

  /**
  Installs a new font
  @throw invalid_argument for nullptr face_

  Exception to be only reported, not handled
  */
  void setFace(FT_Face face_,
               const std::string& fontFile_ /* = ""*/) noexcept(!UT);

  /**
  Enforces sz as vertical size and determines an optimal horizontal size,
  so that most symbols will widen enough to fill more of the drawing square,
  while preserving the designer's placement.

  @param sz desired size of these symbols
  @param bb [out] estimated future bounding box
  @param factorH [out] horizontal scaling factor
  @param factorV [out] vertical scaling factor

  @throw invalid_argument if the parameters prevent the adjustments

  Exception to be only reported, not handled
  */
  void adjustScaling(unsigned sz,
                     FT_BBox& bb,
                     double& factorH,
                     double& factorV) noexcept(!UT);

  PRIVATE :

      /// Symbol settings updating aspect of the Controller
      const IUpdateSymSettings& symSettingsUpdater;

  /// Cmap presenting aspect of the Controller
  const std::unique_ptr<const IPresentCmap>& cmapPresenter;

  /// observer of the symbols' loading, filtering and clustering, who reports
  /// their progress
  AbsJobMonitor* symsMonitor = nullptr;

  FT_Library library = nullptr;  ///< the FreeType lib
  FT_Face face = nullptr;        ///< a loaded font

  VTinySyms tinySyms;  ///< small version of all the symbols from current cmap

  const ISymSettings& ss;  ///< settings of this font engine

  /// indices for each unique Encoding within cmaps array
  boost::bimaps::bimap<FT_Encoding, unsigned> uniqueEncs;

  /// Container with the PixMapSym-s of current charmap
  const std::unique_ptr<IPmsCont> symsCont;

  /// Indices of the symbols that couldn't be loaded
  std::unordered_set<FT_ULong> symsUnableToLoad;

  /// The index of the selected cmap within face's charmaps array
  unsigned encodingIndex = 0U;

  /// Count of glyphs within current charmap (blanks & duplicates included)
  unsigned symsCount = 0U;
};

#endif  // H_FONT_ENGINE
