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

#ifndef H_CLUSTER_SUPPORT
#define H_CLUSTER_SUPPORT

#include "clusterSupportBase.h"

// Forward declarations
struct IClusterProcessing;
struct ISymsSupport;

/**
Polymorphic helper for clustering the symbols depending on the value of PreselectionByTinySyms.
For PreselectionByTinySyms == true, the clusters include tiny symbols.
*/
class ClustersSupport : public IClustersSupport {
protected:
	IClusterProcessing &ce;		///< clusters manager
	const std::uniquePtr<ISymsSupport> ss;	///< helper for symbols
	VSymData &symsSet;		///< set of most information on each symbol

public:
	/// Base constructor
	ClustersSupport(IClusterProcessing &ce_, std::uniquePtr<ISymsSupport> ss_, VSymData &symsSet_);

	ClustersSupport(const ClustersSupport&) = delete;
	ClustersSupport(ClustersSupport&&) = delete;
	void operator=(const ClustersSupport&) = delete;
	void operator=(ClustersSupport&&) = delete;

	/**
	Clusters symsSet. For PreselectionByTinySyms == true it clusters also the tiny symbols.
	@param fontType allows checking for previously conducted clustering of current font type; empty for various unit tests
	*/
	void groupSyms(const std::stringType &fontType = "") override;

	/**
	Rearranges symsSet and its tiny correspondent version when PreselectionByTinySyms == true.
	Computes the cluster representatives and marks the limits between the symbols for different clusters.
	*/
	void delimitGroups(std::vector<std::vector<unsigned>> &symsIndicesPerCluster,
					   VClusterData &clusters, std::set<unsigned> &clusterOffsets) override;

	/// Returns the rearranged symsSet or its tiny correspondent version when PreselectionByTinySyms == true.
	const VSymData& clusteredSyms() const override;
};

#endif // H_CLUSTER_SUPPORT
