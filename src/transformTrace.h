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

#ifndef H_TRANSFORM_TRACE
#define H_TRANSFORM_TRACE

#if defined _DEBUG && !defined UNIT_TESTING

#include <string>
#include <fstream>

struct BestMatch; // forward declaration

/**
Facilitates the tracing process during the transformation of an image.
*/
class TransformTrace {
protected:
	const std::string &studiedCase;	///< used to establish the name of the generated trace file
	const unsigned sz;		///< symbol size
	const bool isUnicode;	///< Unicode symbols are logged in symbol format, while other encodings log just their code

	std::wofstream ofs;				///< trace file stream
	unsigned transformingRow = 0U;	///< the index of the current row being transformed

public:
	/// Opens a trace file stream and initializes required fields
	TransformTrace(const std::string &studiedCase_, unsigned sz_, bool isUnicode_);
	~TransformTrace(); ///< closes the trace stream

	/// adds a new line to the trace file containing row, column and details about the best match for a new patch
	void newEntry(unsigned r, unsigned c, const BestMatch &best);
};


#else // !_DEBUG || UNIT_TESTING

/// Mock class when tracing isn't actually performed 
class TransformTrace {
public:
	TransformTrace(const std::string&, unsigned, bool) {}
	~TransformTrace() {}
	void newEntry(unsigned, unsigned, const BestMatch&) {}
};

#endif // _DEBUG , UNIT_TESTING

#endif // !H_TRANSFORM_TRACE