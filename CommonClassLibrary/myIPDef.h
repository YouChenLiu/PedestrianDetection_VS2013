#include <vector>
using std::vector;

#include <set>
using std::set;

class myBlob;
class myContour;

class mySegmentationImpl;
class mySegmentation;
class mySignature;
class myWatershed;

// background subtraction
class myBGModelImpl;
class myBGModel;
class AbsPixelDifference;
class Stauffer;

/* Class Name: myBlob
*
* Description: implement a class for maintaining a blob
*
* Programmer: poww
* Create Date: 2013.04.01
*/
class myBlob{
	// private attributes
private:
	// a set of points in Blob
	vector<cv::Point> m_vPoint;

	// constructor
public:
	myBlob(void);		// constructor
	~myBlob(void);		// de-constructor

	// public operations
public:
	// get / set operations
	int			GetArea(void);
	cv::Point	GetPoint(int);					// get the blob point at a specific position

	// general operations
	void		AddPoint(cv::Point);			// add the point to the blob

	cv::Rect	CalcBoundingBox(void);			// calculate the bounding box
	cv::Point	CalcCenter(void);				// calculate the blob center

	float		CalcCentralMoment(int, int);	// calculate the moment
	float		CalcDirection(void);			// calculate the direction

	void		Clear(void);					// clear all the elements in the blob

	void		Show(cv::Mat&, cv::Scalar);
};// End of myBlob class


/* Class Name: mySegmentationImpl
*
* Description: an abstract for segmentation implemlentation
*
* Programmer: poww
* Create Date: 2014.08.29
*/
class mySegmentationImpl{
	// protected field
private:
	int			m_iWidth, m_iHeight;	// image dimension

	// constructor
public:
	mySegmentationImpl(int, int);
	~mySegmentationImpl();

	// static operations
public:
	static	int			LabelComponent(cv::Mat&, cv::Mat&);

	// public operations
public:
	// get the size of m_pLabelImage
	cv::Size			GetSize(void);

	virtual void		SetParam(char*, char*) = 0;
	virtual void		SetParamf(int, float) = 0;
	virtual void		SetParami(int, int) = 0;

	// perform segmentation without mask
	virtual int			Segment(cv::Mat&, cv::Mat&, cv::Mat&) = 0;

	// private operations
protected:
};// End of mySegmentationImpl



/* Class Name: mySegmentation
*
* Description: implement a class for segmentation
*
* Programmer: poww
* Create Date: 2014.08.29
*/
class mySegmentation{
	// public constants
public:
	static const int	ID_SIGNATURE = 1;
	static const int	ID_WATERSHED = 2;

	// protected field
private:
	int			m_iWidth, m_iHeight;	// image dimension
	int			m_iBlob;				// number of blobs

	// image field
	cv::Mat		m_LabelImage;		// label image
	cv::Mat		m_ROIMask;			// mask image

	// object field
	mySegmentationImpl*		m_poImpl;

	// constructor
public:
	mySegmentation(int, int);		// construtor
	~mySegmentation(void);			//de-constructor

	// static operations
	static	int		LabelComponent(cv::Mat&, cv::Mat&);

	// public operations
public:
	// get/set operations
	void		GetBlob(vector<myBlob*> &vpBlob);
	void		GetContour(vector<myContour*> &vpContour);

	void		SetParam(char*, char*);
	void		SetParamf(int, float);
	void		SetParami(int, int);

	// factory method
	void	DoMakeImpl(int);		// make segmentation implementation

	// perform segmentation without mask
	int		Segment(cv::Mat&);
	// perform segmentation with mask
	int		Segment(cv::Mat&, cv::Mat&);

	// show images
	void	Show(cv::Mat&, float = 0.5);
};// End of mySegmentation Class


/* Class Name: mySignature
*
* Description: implement signature segmentation algorithm
*
* Programmer: poww
* Create Date: 2015.03.27
*/
class mySignature : public mySegmentationImpl{
	// private fields
private:
	int		m_iParamRowGap;			// available gap in row direction
	int		m_iParamColGap;			// available gap in col direction
	int		m_iParamProjectionTh;	// projection threshold
	int		m_iParamAreaTh;			// area threshold

	vector<int>	m_viColProjection;	// col projection 
	vector<int>	m_viRowProjection;	// row projection

	cv::Mat	m_GrayImage;	// gray Image

	// constructor
public:
	mySignature(int, int);
	~mySignature();

	// public operations
public:
	virtual int		Segment(cv::Mat&, cv::Mat&, cv::Mat&);

	// private operations
private:
	// determine the object range in given projection
	void	DeteObjectRange(vector<int>&, int, int, int, vector<int>&);
	bool	IsContainObject(cv::Mat&, cv::Rect, int);
};// End of mySignature


/* Class Name: myWatershed
*
* Description: implement the watershed segmentation algorithm
*
* Programmer: poww
* Create Date: 2015.04.16
*/
class myWatershed : public mySegmentationImpl{
	// private fields
private:
	float	m_iParamMarkerTh;		// the threshold for maker labeling

	// for gradient computation
	cv::Mat		m_GrayImage;				// for gray image
	cv::Mat		m_IxImage, m_IyImage;		// for image gradient
	cv::Mat		m_ABSIxImage, m_ABSIyImage; // for abs gradient
	cv::Mat		m_ImImage;					// input gradient magnitude

	// for watershed maker
	cv::Mat		m_MarkerMask;				// marker mask
	cv::Mat		m_Marker;					// marker

	// constructor
public:
	myWatershed(int, int);		// constructor with the dimension
	~myWatershed();

	// public operations
public:

	// protected operations
protected:
	// template method
	virtual		int		Segment(cv::Mat&, cv::Mat&, cv::Mat&);

	// to be overloaded by different approaches
	virtual		void	ExtractMarkerMask(cv::Mat&, cv::Mat&, cv::Mat&);
};// End of myWatershed class


/* Class Name: myBGModelImpl
*
* Description: define the interface for background subtraction algorithms
*
* Programmer: poww
* Create Date: 2014.12.10
*/
class myBGModelImpl{
	// private field
private:
	cv::Mat				m_FGMaskImage;	// foreground mask

	// constructor
public:
	myBGModelImpl(int, int);
	~myBGModelImpl(void);

	// public operations
public:
	virtual void	SetParam(char*, char*) = 0;
	virtual void	SetParamf(int, float) = 0;
	virtual void	SetParami(int, int) = 0;

	virtual void	Model(vector<cv::Mat>&) = 0;
	virtual int		Segment(cv::Mat&, cv::Mat&, cv::Mat&);
	virtual void	Subtract(cv::Mat&, cv::Mat&) = 0;
	virtual void	Update(cv::Mat&) = 0;
};// End of myBGModelgImpl class


/* Class Name: myBGModel
*
* Description: implement a class for background subtraction
*
* Programmer: poww
* Create Date: 2014.12.10
*/
class myBGModel : public mySegmentation{
	// public constants
public:
	static const int	ID_ABSPIXELDIFFERENCE = 1;
	static const int	ID_Stauffer = 2;

	// private field
private:
	cv::Mat				m_FGMaskImage;	// foreground mask

	// constructor
public:
	myBGModel(int, int);	// constructor with image dimension
	~myBGModel(void);

	// public operations
public:
	void		DoMakeImpl(int);			// factory method	for creating implementation

	void		Model(cv::Mat&);			// model background
	void		Model(vector<cv::Mat>&);	// model background

	void 		Subtract(cv::Mat&, cv::Mat&);			// perform subtraction

	void		Update(cv::Mat&);			// update the background model
};// End of myBGModel


/* Class Name: AbsPixelDifference
*
* Description: implement a class for background subtraction using direct pixel difference
*
* Programmer: poww
* Create Date: 2015.04.28
*/
class AbsPixelDifference : public myBGModelImpl{
	// public constants
public:
	static const int	PARAM_ALPHA = 1;
	static const int	PARAM_DTh = 2;

	// private fields
private:
	int			m_iParamDTh;			// difference threshold
	float		m_fParamAlpha;			// learning rate

	// image fields
	cv::Mat		m_BGImage;

	// constructor
public:
	AbsPixelDifference(int, int, int = 20, float = 0.01);	// constructor with image dimension
	~AbsPixelDifference();

	// public operations
public:
	// get / set operations
	virtual void	SetParam(char*, char*){};
	virtual void	SetParami(int, int);
	virtual void	SetParamf(int, float);

	virtual void	Model(vector<cv::Mat>&);
	virtual void	Subtract(cv::Mat&, cv::Mat&);
	virtual void	Update(cv::Mat&);
};// End of AbsPixelDifference


/**
* class Stauffer
*
*	implement the background subtraction algorithm proposed by C. Stauffer in CVPR 1999
*
* Create Date: 2004.07.13
* Modify Date: 2006.02.16
* Modify Date: 2015.04.17: Use Opencv Library 2.4.11 with cv::Mat structure
*
* Programmer: poww
**/
class Stauffer : public myBGModelImpl{
	// public constants
public:
	static const int	ID_PARAM_ALPHA = 1;
	static const int	ID_PARAM_K = 2;
	static const int	ID_PARAM_T = 3;
	static const int	ID_PARAM_MATCHTH = 4;

	// private constants
private:
	static float	DEF_VAR[3];// = {1000.0, 1000.0, 1000.0};
	static float	DEF_WEIGHT;	// default weight = 0.05

	/* private attributes */
private:
	/* attributes */
	int				m_iWidth, m_iHeight;		// image dimension
	int				m_iK;						// number of Gaussian

	float			m_fAlpha, m_fT;				// model parameters
	float			m_fParamMatchTh;					// determing the match of a Gaussian 

	myMixGaussian**	m_ppoMixGaussian;

	// constructor
public:
	Stauffer(int, int, int = 3, float = 0.01, float = 0.7);
	~Stauffer(void);

	/* public operations */
public:
	// get / set operations
	virtual void	SetParam(char*, char*){};
	virtual void	SetParami(int, int);
	virtual void	SetParamf(int, float);

	virtual void	Model(vector<cv::Mat>&);
	virtual void	Subtract(cv::Mat&, cv::Mat&);
	virtual void	Update(cv::Mat&);

	// private operations
private:
	void	AllocateModel(myMixGaussian**, int);
};// End of Stauffer class
