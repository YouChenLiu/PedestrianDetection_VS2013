#include <vector>
using std::vector;

#include <set>
using std::set;

class myString;

class myFile;
class myImReaderInterface;
class myImWriterInterface;
class myTextRWInterface;

class myImSequence;

// math 
class myGaussian;
class myMixGaussian;
class myDisjointSet;

/* Class Name: myString
*
* Description: maintain a set of string operations widely used for image processing
*
* Programmer: poww
* Create Date: 2013.05.15
*/
class myString{
	// constructor
public:
	myString(void);
	~myString(void);
	// static public operations
public:
	static void Concatenate(char*, int, int, ...);					// concatenate a list of strings 
	static void Expand(int, int, int, int, vector<char*>&);		// expand the iterative expression to a list of strings		
	static void	Split(char*, char, vector<char*>&);
	static void ValueOf(int, int, char*);
};// End of myString class



/* Class Name: myFile
*
* Description: It defines the interface for file opening and closing
*
* Programmer: poww
* Create Date: 2014.0812
*/
class myFile{
	// public constants
public:
	// define the parameter identifier
	static const int READ = 0;		// read mode
	static const int WRITE = 1;		// write mode

	// protected attributes
protected:
	int				m_iMode;			// mode for reading / writing

	// constructor 
public:
	myFile(void);
	~myFile(void);

	// public virtual operations
public:
	virtual void	Close(void) = 0;			/* close a file */
	virtual bool	Open(char*, int) = 0;		/* open a file*/
};// End of myFile class


/* Class Name: myImReaderInterface
*
* Description: define an interface for image reading
*
* Programmer: poww
* Create Date: 2014.0812
*/
class myImReaderInterface{
	// constructor
public:
	myImReaderInterface(void);
	~myImReaderInterface(void);

	// public
public:
	virtual bool ReadImage(cv::Mat&) = 0;
};// End of myImReaderInterface class

/* Class Name: myImWriterInterface
*
* Description: define an interface for image writing
*
* Programmer: poww
* Create Date: 2014.12.16
*/
class myImWriterInterface{
	// constructor
	public:
		myImWriterInterface();
		~myImWriterInterface();

	// publicB
	public:
		virtual bool WriteImage(cv::Mat&) = 0;
};// End of myImWriterInterface


/* Class Name: myTextRWInterface
*
* Description: define an interface for text writing and reading
*
* Programmer: poww
* Create Date: 2014.0812
*/
class myTextRWInterface{
	// constructor
	public:
		myTextRWInterface();
		~myTextRWInterface();

	// public
	public:
		virtual bool ReadString(char*, int) = 0;
		virtual bool WriteString(char*, int) = 0;
};// End of myTextRWInterface

/* Class Name: myAVIFile
*
* Description: It implmenets the functions for manipulating the AVI file
*
* Programmer: poww
* Create Date: 2013.05.19
*/
class myAVIFile : myFile, myImReaderInterface, myImWriterInterface{
	// private attributes
private:
	int				m_iWidth, m_iHeight;// image resolution
	int				m_iFPS;				// frame per second

	cv::VideoCapture*	m_pCapture;
	cv::VideoWriter*	m_pWriter;

	// constructor
public:
	myAVIFile();
	~myAVIFile();

	// public operations
public:
	virtual void	Close(void);				// close a file
	virtual bool	Open(char*, int);			// open a file
	bool			Open(char*, int, int, int);	// open for writing with the specific writing format

	virtual bool	ReadImage(cv::Mat&);	// read an image from the avi file
	virtual bool	WriteImage(cv::Mat&);				// write the image to the avi file
};// End of myAVIFile class


/**
* Class Name: myImSequence
*
* Description: implement functionality to read a sequence of images
*
* Programmer: poww
* Create Date: 2014.10.20
*/
class myImSequence : public myImReaderInterface, myImWriterInterface{
	// public constants
public:
	static const int INDEX = 1;
	static const int LENGTH = 2;
	static const int STEP = 3;

	// fields
private:
	char	m_sExtension[10];
	char	m_sPrefix[100];

	int		m_iSeqIndex;
	int		m_iSeqLength;
	int		m_iSeqStep;

	// constructor
public:
	myImSequence(char*, char*);		// constructor with prefix and extension
	~myImSequence(void);

	// set operations
public:
	void	SetParami(int, int);	// set the parameters with id and its value

	// public operations
public:
	virtual bool ReadImage(cv::Mat&);
	virtual bool WriteImage(cv::Mat&);
};// End of myImSequence class


/**
 *		Math Module
 **/

/* Class Name: Gaussian
*
* Description: maintain a 1D Gaussian distribution
*
* Programmer: poww
* Create Date: 2012.10.02
* Version: v010
*/
class myGaussian{
	// private attributes
	int		m_iDim;		// dimension
	float*	m_pfMean;	// mean vector
	float*	m_pfVar;	// sigma vector

	// constructor
public:
	myGaussian(int);		// with dimension
	~myGaussian(void);

	// public operation
public:
	// static operations
	static void CalcMean(vector<float*>&, int, float*);
	static void CalcVar(vector<float*>&, int, float*);

	// get / set operations
	float*	GetMean(void);				// get the mean vector
	float*	GetVar(void);				// get the variance vector

	void	SetMean(float*);			// set the mean vector
	void	SetVar(float*);				// set the variance vector

	float	CalcDeterminant(void);		// calculate the Determinant
	float	CalcMahalanobis(float*);	// calculate the distance to the mean
	float	CalcPr(float*);				// calculate the probability

	void	Form(vector<float*>&);		// form the probability

	void	Update(float*, float);		// update the probability
};// End of myGaussina class

/* Class Name: myMixGaussian
*
* Description: maintain mixture of K Gaussians
*
* Programmer: poww
* Create Date: 2012.10.02
* Version: v010
*/
class myMixGaussian{
	// private attributes
private:
	int		m_iK;

	vector<float>		m_vfWeight;		// a list of Gaussian weight
	vector<myGaussian*>	m_vpoGaussian;	// a list of Gaussian object references

	// constructors
public:
	myMixGaussian(int, int); // with k and feature dimension
	~myMixGaussian(void);

	// public operations
public:
	// get / set operations
	myGaussian*  GetGaussian(int);	// get the kth Gaussian
	float	GetWeight(int);			// get the weight of kth Gaussian
	void	SetWeight(int, float);	// set the weight of the kth Gaussian

	// public operations
	int		ArgMaxWeight(void);		// find the Gaussian index with max weight
	int		ArgMinWeight(void);		// find the Gaussian index with min weight

	myGaussian*		MaxWeight(void);	// find the Gaussian reference with max weight
	myGaussian*		MinWeight(void);	// find the Gaussian reference with min weight

	void	Normalize(void);		// perform weight normalization
	void	Sort(void);				// perform the sorting according to weight/variance
};// End of myMixGaussian class


/* myDisjointSet
*
* Description:
*		maintain a forest of disjoint sets
*
* Programmer: poww
* Create Date: 2014.08.22
*/
class myDisjointSet{
	// private attributes
private:
	typedef struct myKeySet{
		set<int>	aSet;
		int			key;
	} myKetSet;

	vector<myKeySet*>	m_vpKeySet;		// a vector of set pointer

	// constructor
public:
	myDisjointSet(void);			// constructor
	~myDisjointSet(void);			// de-constructor

	// public operations
public:
	// get / set operations
	// get member of a specific set
	void	GetMemberOfSet(int, vector<int>&);
	int		GetNumOfSet(void);		// return number of sets

	void	Clear(void);		// clear the myDisjointSet

	int		Find(int);			// find the setID

	bool	IsMember(int);		// check if an element is the mber of myDisjointSet

	void	MakeSet(int, ...);	// make a set containing given elements

	void	Show(void);			/// show the disjoint count

	void	Union(int, int);	// union the sets containing the input elements
};// End of myDisjointSet


