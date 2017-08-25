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

#ifndef H_MATCH_PROGRESS
#define H_MATCH_PROGRESS

/**
Type for a polymorphic parameter to be passed to MatchEngine::improvesBasedOnBatch.
The type is adjusted based on the value of PreselectionByTinySyms.
Each time the method finds a better match, it calls remarkedMatch from this class.

This is intentionally not abstract, with an empty implementation of remarkedMatch,
to allow providing dummy instances as parameter placeholder (for PreselectionByTinySyms == false).
*/
class MatchProgress {
public:
	/// Base class constructor
	MatchProgress() {}

	MatchProgress(const MatchProgress&) = delete;
	MatchProgress(MatchProgress&&) = delete;
	void operator=(const MatchProgress&) = delete;
	void operator=(MatchProgress&&) = delete;

	virtual ~MatchProgress() {}

	/**
	For PreselectionByTinySyms == false, it does nothing.
	For PreselectionByTinySyms == true, it checks if the new match is worth placing on the short candidate list.
	*/
	virtual void remarkedMatch(unsigned /*symIdx*/, double /*score*/) {}
};

#endif // H_MATCH_PROGRESS
