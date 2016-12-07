/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#ifndef H_PROPS_READER
#define H_PROPS_READER

#pragma warning ( push, 0 )

#include <boost/filesystem/path.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#pragma warning ( pop )

/**
Parser for reading mandatory properties from a configuration file.
Besides wrapping 'info_parser', it adds error messages to any exceptions.
*/
class PropsReader {
	PropsReader(const PropsReader&) = delete;
	void operator=(const PropsReader&) = delete;

protected:
	const boost::filesystem::path propsFile;	///< path to the configuration file
	boost::property_tree::ptree props;			///< the property tree built from the configuration

public:
	/**
	Builds the parser.

	@param propsFile_ the path to the configuration file

	@throw info_parser_error when the file doesn't exist or cannot be parsed
	*/
	PropsReader(const boost::filesystem::path &propsFile_);

	/**
	Reads a certain property assuming it has type T.

	@param prop the name of the property to be read

	@throw ptree_bad_path when prop is not a valid property or it hasn't be found within the configuration
	@throw ptree_bad_data when prop exists, but it cannot be converted to type T
	*/
	template<typename T>
	T read(const std::string &prop) const {
		try {
			return std::move(props.get<T>(prop));
		} catch(boost::property_tree::ptree_bad_path&) {
			cerr<<"Property '"<<prop<<"' is missing from '"<<propsFile<<"' !"<<endl;
		} catch(boost::property_tree::ptree_bad_data&) {
			cerr<<"Property '"<<prop<<"' cannot be converted to its required type!"<<endl;
		}
		throw;
	}
};

#endif