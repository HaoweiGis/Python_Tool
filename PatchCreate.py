from osgeo import gdal,ogr,osr
import numpy as np
import matplotlib.pyplot  as plt
import os,sys
from skimage.measure import label
from scipy import optimize
from sklearn.linear_model import Lasso
import lmfit
# from piecewise.regressor import piecewise
# from piecewise.plotter import plot_data_with_regression

class Light():
    def __init__(self,filename,citycode):
        # self.path = os.path.join(os.getcwd(),'CodePython')
        self.filename = filename
        self.path = os.getcwd()
        self.max_index = 80
        self.citycode = citycode
        self.citypath = os.path.join(self.path,'Input\\'+citycode)
        filetiff = os.path.join(self.citypath,r'StatusData\\Original\Status_1.tif')
        # self.start = 5
        self.start = 5
        self.image1 = np.array(gdal.Open(filetiff).ReadAsArray(), dtype='int16')
        self.maxdn = np.max(self.image1)
        # print('maxdn:',self.maxdn)
        # print(self.citypath)
        
    def GeotiffR(self):
        dataset = gdal.Open(self.filename)
        im_porj = dataset.GetProjection()
        im_geotrans = dataset.GetGeoTransform()
        im_data = np.array(dataset.ReadAsArray(), dtype='int32')
        im_shape = im_data.shape
        # print(im_geotrans)
        # print('zuobiaoxi',im_porj)
        del dataset
        return im_data,im_porj,im_geotrans,im_shape

    def GeotiffW(self,filename,im_shape,single,im_geotrans,im_porj):
        datatype = gdal.GDT_Int32
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename,im_shape[1],im_shape[0],1, datatype)
        dataset.SetGeoTransform(im_geotrans)             
        dataset.SetProjection(im_porj)
        dataset.GetRasterBand(1).WriteArray(single)
        del dataset  

    def RGeotiff(self, filename):
        sourceRaster = gdal.Open(filename)
        band = sourceRaster.GetRasterBand(1)
        porj = sourceRaster.GetProjectionRef()
        return sourceRaster,band,porj

    def RasterToVector(self,filetiff,filemask,fileshp):
        sourceRaster,band,porj = self.RGeotiff(filetiff)
        sourceRasterm,bandm,porjm = self.RGeotiff(filemask)
        outShapefile = fileshp+'.shp'
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outShapefile):
            driver.DeleteDataSource(outShapefile)
        outDatasource = driver.CreateDataSource(outShapefile)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(porj)
        outLayer = outDatasource.CreateLayer("polygonized", srs = srs)
        oFieldID = ogr.FieldDefn('DN',ogr.OFTInteger)
        outLayer.CreateField(oFieldID, 1)
        gdal.Polygonize( band, bandm, outLayer,0 , [], callback=None )
        # gdal.Polygonize( band,None,outLayer)
        outDatasource.Destroy()
        sourceRaster = None
        sourceRasterm = None

    # Generate patch , patch_mask and original
    def BatchCreate(self,im_data):
        im_ntl = None
        im_original = None
        for status in range(1,self.max_index+1):
            ntl_class=np.zeros(im_data.shape)
            ntl_original = np.zeros(im_data.shape)
            indexv = np.where(im_data>=status)
            ntl_original[indexv]= im_data[indexv]
            ntl_class[indexv]= 1
            ntl_class = label(ntl_class, connectivity = 1)
            if im_ntl is None:
                im_ntl = ntl_class
                im_original = ntl_original
            else:   
                im_ntl = np.concatenate((im_ntl, ntl_class), axis=0)     
                im_original = np.concatenate((im_original,ntl_original),axis = 0)
        # im_ntl = im_ntl.astype(int)
        im_mask=np.zeros(im_ntl.shape)
        indexnull = np.where(im_ntl != 0)
        im_mask[indexnull] = 1
        # print(im_ntl)
        return im_ntl,im_mask,im_original

    # patch file generate 
    def BatchGenerate(self):
        im_data,im_porj,im_geotrans,im_shape = self.GeotiffR()
        im_ntl,im_mask,im_original = self.BatchCreate(im_data)
        for i in range(self.max_index):
            index1 = im_shape[0]*i
            index2 = im_shape[0]*(int(i)+1)
            # print(index1,index2)
            singlentl = im_ntl[index1:index2,:]  
            singlemask = im_mask[index1:index2,:] 
            singleOriginal = im_original[index1:index2,:]
            # print(single.shape)
            # create file path 
            filetiff = os.path.join(self.citypath,r'StatusData\Status_'+str(i+1)+'.tif')
            fileOrig = os.path.join(self.citypath,r'StatusData\\Original\Status_'+str(i+1)+'.tif')
            fileshp = os.path.join(self.citypath,r'StatusData\Shape\Status_'+str(i+1))
            filemask = os.path.join(self.citypath,r'StatusData\Shape\Mask_'+str(i+1)+'.tif')
            print(filetiff)             
            self.GeotiffW(filetiff,im_shape,singlentl,im_geotrans,im_porj)
            self.GeotiffW(filemask,im_shape,singlemask,im_geotrans,im_porj)
            self.GeotiffW(fileOrig,im_shape,singleOriginal,im_geotrans,im_porj)
            self.RasterToVector(filetiff,filemask,fileshp)         

    def CalcDN(self):
        CaArray , MaxdnArray = [],[]
        filetiff = os.path.join(r'E:\TeacherZ\NTL_Landscape\CodePython\Data\citytif',str(self.citycode)+'.tif')
        cityimage = np.array(gdal.Open(filetiff).ReadAsArray(), dtype='float')
        cityArea = len(np.where(cityimage >= 0)[0])
        CA = cityArea*0.25
        CaArray.append(CA)
        # print(cityArea)
        cityDNAll = []
        rasterDNAll = []
        for i in range(self.start,self.max_index+1):
            filetiff = os.path.join(self.citypath,r'StatusData\\Original\Status_'+str(i)+'.tif')
            im_data = np.array(gdal.Open(filetiff).ReadAsArray(), dtype='int16')
            
            if i <= self.maxdn:
                rasterArea = len(np.where(im_data!=0)[0])         
                # print(rasterArea)
                cityDN = np.sum(im_data)/cityArea
                rasterDN = np.sum(im_data)/rasterArea
                # print(cityDN,rasterDN)
            else:
                cityDN,rasterDN = 0,0
            cityDNAll.append(cityDN)
            rasterDNAll.append(rasterDN)
        Maxdn = self.maxdn   
        MaxdnArray.append(Maxdn)
        # print(len(cityDNAll),len(rasterDNAll))
        return CaArray,MaxdnArray,cityDNAll,rasterDNAll

    def CalcSingle(self):
        PatchNum =[]
        PatchSumAll =[]
        MpaAll = []
        MpsAll = []
        SiAll = []
        CovAll = []
        CiAveAll = []
        CiRaAll = []
        CiSdAll = []
        LpiAll = []
        PdAll = []
        DivisionAll = []
        for i in range(self.start,self.max_index+1):
            filetiff = os.path.join(self.citypath,r'StatusData\Status_'+str(i)+'.tif')
            im_data = np.array(gdal.Open(filetiff).ReadAsArray(), dtype='int16')
            Pnum = len(np.unique(im_data[np.where(im_data!=0)]))
            # print(Pnum)
            if Pnum != 0:
                Arealist = []
                for j in range(1,Pnum+1):
                    index = np.where(im_data==j)
                    Area = len(im_data[index])*0.25
                    Arealist.append(Area)
                # print(Arealist)
                # break
                PatchSum = np.sum(Arealist) #总面积
                Division = np.subtract(1,np.sum(np.power(np.divide(Arealist,PatchSum),2))) #景观分离度
                # print(Division)
                Mpa = np.max(Arealist) #最大面积
                Mps = np.divide(np.sum(Arealist),Pnum) #平均面积
                Si = np.around(np.sqrt(np.mean(np.power(Arealist-Mps,2))),4) #面积标准差
                # print(Si)
                Cov = np.around(np.divide(Si,np.mean(Arealist)),4) #变异系数
                Ci = np.divide(Si,Arealist) #变动系数 数组  
                CiAve,CiRa,CiSd = self.CalcMean(Ci)
                Lpi = np.around(Mpa/PatchSum,4) #最大斑块指数
                Pd = np.around(Pnum/PatchSum,4) #破碎度 斑块密度
                # print(Division)
            else:
                Pnum,PatchSum,Mpa,Mps,Si,Cov,CiAve,CiRa,CiSd,Lpi,Pd,Division = 0,0,0,0,0,0,0,0,0,0,0,0
            
            PatchNum.append(Pnum)

            PatchSumAll.append(PatchSum)
            DivisionAll.append(Division)
            MpaAll.append(Mpa)
            MpsAll.append(Mps)
            SiAll.append(Si)
            CovAll.append(Cov)
            CiAveAll.append(CiAve)
            CiRaAll.append(CiRa)
            CiSdAll.append(CiSd)
            LpiAll.append(Lpi)
            PdAll.append(Pd)
        # print(len(DivisionAll))
        return PatchNum,PatchSumAll,MpaAll,MpsAll,SiAll,CovAll,CiAveAll,CiRaAll,CiSdAll,LpiAll,PdAll,DivisionAll

    def CalcMean(self,IndexAll):
        Ave = np.around(np.divide(np.sum(IndexAll),len(IndexAll)),4)
        Ra = np.around(np.subtract(np.max(IndexAll),np.min(IndexAll)),4)
        Sd = np.around(np.sqrt(np.divide(np.sum(np.power(np.subtract(IndexAll,np.mean(IndexAll)),2)),len(IndexAll))),4)    
        return Ave,Ra,Sd #均值，极差，标准差

    def CalcVectorIndex(self):
        FracAveAll = []
        FracRaAll = []
        FracSdAll = []
        ConhesionAll = []
        # Cfile = os.path.join(r'E:\TeacherZ\NTL_Landscape\CodePython\Data\shape',str(self.citycode)+'.shp')
        # Cdataset = ogr.Open(Cfile)
        # # print(Cfile)
        # CoLayer = Cdataset.GetLayer()
        # Cfeature = CoLayer.GetFeature()
        # Cgeometry = Cfeature.GetGeometryRef()
        # CPerimeter = np.around(Cgeometry.Boundary().Length(),2)
        # print(CPerimeter)
        print(self.maxdn)
        for i in range(self.start,self.max_index+1):
            fileshp = os.path.join(self.citypath,r'StatusData\Shape\Status_'+str(i)+'.shp')
            # print(fileshp)
            dataset = ogr.Open(fileshp)
            oLayer = dataset.GetLayerByIndex(0)
            AreaAll = []
            PerimeterAll = []
            # if oLayer.GetFeatureCount(0) != 0:
            if i < self.maxdn:
            # print(oLayer.GetFeatureCount())
                for index in range(oLayer.GetFeatureCount(0)):
                    feature = oLayer.GetFeature(index)
                    geometry = feature.GetGeometryRef()
                    # get the area
                    Area = np.around(geometry.Simplify(300).GetArea(),2)
                    Perimeter = np.around(geometry.Simplify(300).Boundary().Length(),2)
                    PerimeterAll.append(Perimeter)
                    AreaAll.append(Area)
                    # print(Area,Perimeter)
                    Con1 = np.subtract(1,np.divide(np.sum(PerimeterAll),np.sum(np.multiply(PerimeterAll,np.sqrt(AreaAll)))))
                    Con2 = np.subtract(1,np.divide(1,np.sqrt(np.sum(AreaAll))))
                    Conhesion = np.multiply(Con1,Con2)*100
                    # print(Conhesion)
                    FracAll = np.divide(np.log(np.divide(PerimeterAll,4)),np.log(AreaAll))*4
                    FracAve,FracRa,FracSd = self.CalcMean(FracAll) #分维数
            else:  
                Area = 0 
                Perimeter = 0
                Conhesion = 0
                AreaAll.append(Area)
                PerimeterAll.append(Perimeter)
                FracAve,FracRa,FracSd = 0,0,0
            # Lsi = 
            FracAveAll.append(FracAve)
            FracRaAll.append(FracRa)
            FracSdAll.append(FracSd)
            ConhesionAll.append(Conhesion)
        # print(ConhesionAll)
        return FracAveAll,FracRaAll,FracSdAll,ConhesionAll

    def CalcIndex(self):
        CutRateAll = []
        CutNumAll = []
        DiedAll = []
        UnchangeAll = []
        ShrinkAll = []
        SplitAll = []
        for i in range(self.start,self.max_index):#i image index
            if i < self.maxdn:
                filetiff1 = os.path.join(self.citypath,r'StatusData\Status_'+str(i)+'.tif')
                filetiff2 = os.path.join(self.citypath,r'StatusData\Status_'+str(i+1)+'.tif')
                im_data1 = np.array(gdal.Open(filetiff1).ReadAsArray(), dtype='int16')
                im_data2 = np.array(gdal.Open(filetiff2).ReadAsArray(), dtype='int16')
                num_p_1=len(np.unique(im_data1))
                num_p_2=len(np.unique(im_data2))
                Died = 0
                Unchange = 0
                Shrink = 0
                Split = 0
                areasum1 = len(np.where(im_data1!=0)[0])
                areasum2 = len(np.where(im_data2!=0)[0])
                CutRate = np.divide(areasum2,areasum1)
                CutNum = np.subtract(areasum2,areasum1)
                # print(arearate)
                for j in range(1,num_p_1):   #j patch        
                    index = np.where(im_data1==j)
                    id_p2=np.unique(im_data2[index]) #代表2图像中的斑块数
                    # print(id_p2)
                    if len(id_p2)==1 and id_p2==0: #如果现在图像中只有一个值且为0，那么原斑块消失，所以state记录原图像状态
                        Died = Died +1   #消亡 
                    elif len(id_p2)==1 and id_p2 !=0:
                        Unchange = Unchange + 1  #未变
                    elif len(id_p2)==2 and sum(id_p2==0)==1: 
                        Shrink = Shrink + 1   #收缩
                    elif len(id_p2)>=2 :
                        Split = Split + 1   #分裂        
                # Line = str(Died)+','+str(Unchange)+','+str(Shrink)+','+str(Split)
            else:
                CutRate = 0 
                CutNum = 0
                Died = 0
                Unchange = 0
                Shrink = 0
                Split = 0
            CutRateAll.append(CutRate)
            CutNumAll.append(CutNum)
            DiedAll.append(Died)
            UnchangeAll.append(Unchange)
            ShrinkAll.append(Shrink)
            SplitAll.append(Split)
        # print(len(CutRateAll))
        return CutRateAll,CutNumAll,DiedAll,UnchangeAll,ShrinkAll,SplitAll

    def CalcSDE(self):
        SDEArea = []
        SDEFlat = []
        SDETan = []
        for i in range(self.start,self.max_index+1):#i image index
            # print(i,self.maxdn)
            if i < self.maxdn:
                filetiff = os.path.join(self.citypath,r'StatusData\Status_'+str(i)+'.tif')
                # print(filetiff)
                im_data = np.array(gdal.Open(filetiff).ReadAsArray(), dtype='int16')
                # print(im_data.shape)
                X = np.where(im_data>0)[0]
                Y = np.where(im_data>0)[1]
                # print(X)
                SDEx = np.sqrt(np.sum(np.square(X-np.mean(X))/len(X)))
                # print(SDEx)
                SDEy = np.sqrt(np.sum(np.square(Y-np.mean(Y))/len(Y)))
                # print(SDEx)
                Xmean = X - np.mean(SDEx)
                Ymean = Y - np.mean(SDEy)
                A = np.subtract(np.sum(np.square(Xmean)),np.sum(np.square(Ymean)))
                C = np.sum(np.multiply(Xmean,Ymean))*2
                B = np.sqrt(np.add(np.square(A),np.square(C)))
                tan = np.around(np.divide(np.add(A,B),C),2) #方向
                angle = np.arctan(tan)
                axisx = np.sum(np.square(np.subtract(np.multiply(Xmean,np.cos(angle)),np.multiply(Ymean,np.sin(angle)))))
                axisy = np.sum(np.square(np.add(np.multiply(Xmean,np.sin(angle)),np.multiply(Ymean,np.cos(angle)))))
                sigmoidx = np.multiply(np.sqrt(2),np.sqrt(np.divide(axisx,len(Xmean))))
                sigmoidy = np.multiply(np.sqrt(2),np.sqrt(np.divide(axisy,len(Ymean))))
                # print(sigmoidx,sigmoidy)
                area = np.around(np.pi*sigmoidx*sigmoidy/4,2) #面积
                flat = np.around(sigmoidx/sigmoidy,2) #扁率
                # x = sigmoidx*np.cos(angle)
                # y = sigmoidy*np.sin(angle)
            else:
                tan ,area, flat = 0,0,0
            SDETan.append(tan)
            SDEArea.append(area)
            SDEFlat.append(flat)
        # print(SDEArea)
        return SDEArea,SDEFlat,SDETan

    def BatchTrain(self,pltdata,xvalue,ylabel):
        x = np.linspace(self.start,xvalue,xvalue-self.start+1)
        # print(x)
        # print(len(x))
        y = np.array(pltdata)
        # print(pltdata) 
        # y = np.divide(np.subtract(y,np.min(y)),np.subtract(np.max(y),np.min(y)))
        filename = str(self.start) + '_' + str(self.max_index)+ylabel+'.csv'
        # self.CSVWrite(filename,np.around(y,4),self.citycode)

        def err(w):
            th0 = w['th0'].value
            th1 = w['th1'].value
            th2 = w['th2'].value
            gamma = w['gamma'].value
            fit = th0 + th1*x + th2*np.maximum(0,x-gamma)
            return fit-y
            
        p = lmfit.Parameters()
        p.add_many(('th0', 0.), ('th1', 0.0),('th2', 0.0),('gamma', 40.))
        mi = lmfit.minimize(err, p)
        b0 = mi.params['th0']; b1=mi.params['th1'];b2=mi.params['th2']
        gamma = int(mi.params['gamma'].value)
        
        X0 = np.array(range(self.start,gamma+1,1))
        X1 = np.array(range(0,xvalue+1-gamma,1))
        y0 = y[:gamma-self.start+1]
        y1 = y[gamma-self.start:]
        Y0 = b0 + b1*X0
        Y1 = (b0 + b1 * float(gamma) + (b1 + b2)* X1)
        X = np.append(X0,X1+gamma)
        Y = np.append(Y0,Y1)
        # if gamma > self.start:
        # print(gamma)
        # print('value',y)
        xz1 = int(gamma-((gamma-self.start)/2)-self.start)
        xz2 = int(gamma+((self.max_index-gamma)/2)-self.start)
        # print(xz1,xz2,gamma)
        R_square0 = np.subtract(1,np.divide(np.sum(np.power(np.subtract(y0,Y0),2)),\
        np.sum(np.power(np.subtract(y0,np.average(y0)),2))))
        R_square1 = np.subtract(1,np.divide(np.sum(np.power(np.subtract(y1,Y1),2)),\
        np.sum(np.power(np.subtract(y1,np.average(y1)),2))))
        # print(R_square0,R_square1)
        # print(X,Y)
        plt.scatter(x, y,s =8)
        plt.plot(X0[-1],Y0[-1],'ks')
        plt.vlines(X0[-1],np.min(y),Y0[-1], colors = "c", linestyles = "dashed")
        plt.vlines(X[xz1],np.min(y),Y[xz1], colors = "0.5", linestyles = "dashed")
        plt.vlines(X[xz2],np.min(y),Y[xz2], colors = "0.5", linestyles = "dashed")
        plt.ylim( np.min(y), )
        plt.plot(X,Y,color='r')
        plt.plot(X0,Y0,color='b',label = r'$y0=%s + %sx,R^2 =%s$' %(np.around(b0,2),np.around(b1,2),np.around(R_square0,decimals=2)))
        plt.plot(X1+gamma,Y1,color='r',label = r'$y1=%s+%sx+%s(x-%s),R^2 =%s$'%(np.around(b0,2),np.around(b1,2),np.around(b2,2),gamma,np.around(R_square1,decimals=2)))
        plt.xlabel('Radiance Value'+'(nW '+r'$\mathbf{\mathrm{cm^{-2}}}$ '+ r'$sr^{-1}$)')
        plt.ylabel(ylabel)
        font1 = {'family' : 'Times New Roman',  
            'weight' : 'normal',  
            'size'   : 10,  
            }   
        legend = plt.legend(prop=font1)#loc='upper right',
        frame = legend.get_frame() 
        frame.set_alpha(1) 
        frame.set_facecolor('none')
        # plt.rc('font',family='Times New Roman') 
        pathfig = os.path.join(self.citypath,ylabel+'.jpg')
        plt.savefig(pathfig)
        plt.close()
        # # print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS',ylabel)
        # # print(gamma)
        # # print(x)
        # # print(X)
        # # print(y)
        # # print(Y)
        return gamma,x,y,X,Y

    def BatchPlot(self,parm1,parm2,parm3,parm4,parm5,parm6,parm7,parm8,parm9,parm10,parm11,parm12,parm13,parm14,parm15,parm16,\
        parm17,parm18,parm19,parm20,parm21,parm22,parm23,parm24,parm25,parm26,parm27):
        plt.figure()
        # PatchNum,LpiAll,PdAll,DivisionAll,ConhesionAll,CutRateAll,SDEFlat,SDETan
        # cityDNAll,rasterDNAll,PatchNum,PatchSumAll,MpaAll,MpsAll,SiAll,CovAll,\
        # CiAveAll,CiRaAll,CiSdAll,LpiAll,PdAll,DivisionAll,FracAveAll,FracRaAll,FracSdAll,ConhesionAll,\
        # gamma1,x1,y1,X1,Y1 = self.BatchTrain(parm1,self.max_index,'cityDNAll')
        # gamma2,x2,y2,X2,Y2 = self.BatchTrain(parm2,self.max_index,'rasterDNAll')
        gamma3,x3,y3,X3,Y3 = self.BatchTrain(parm3,self.max_index,'Number of patch')
        # gamma4,x4,y4,X4,Y4 = self.BatchTrain(parm4,self.max_index,'PatchSumAll')
        # gamma5,x5,y5,X5,Y5 = self.BatchTrain(parm5,self.max_index,'MpaAll')
        # gamma6,x6,y6,X6,Y6 = self.BatchTrain(parm6,self.max_index,'MpsAll')
        # gamma7,x7,y7,X7,Y7 = self.BatchTrain(parm7,self.max_index,'SiAll')
        # gamma8,x8,y8,X8,Y8 = self.BatchTrain(parm8,self.max_index,'CovAll')
        # gamma9,x9,y9,X9,Y9 = self.BatchTrain(parm9,self.max_index,'CiAveAll')
        # gamma10,x10,y10,X10,Y10 = self.BatchTrain(parm10,self.max_index,'CiRaAll')
        # gamma11,x11,y11,X11,Y11 = self.BatchTrain(parm11,self.max_index,'CiSdAll')
        gamma12,x12,y12,X12,Y12 = self.BatchTrain(parm12,self.max_index,'Maximum patch index')
        gamma13,x13,y13,X13,Y13 = self.BatchTrain(parm13,self.max_index,'patch density')
        gamma14,x14,y14,X14,Y14 = self.BatchTrain(parm14,self.max_index,'patch division')
        # gamma15,x15,y15,X15,Y15 = self.BatchTrain(parm15,self.max_index,'FracAveAll')
        # gamma16,x16,y16,X16,Y16 = self.BatchTrain(parm16,self.max_index,'FracRaAll')
        # gamma17,x17,y17,X17,Y17 = self.BatchTrain(parm17,self.max_index,'FracSdAll')
        gamma18,x18,y18,X18,Y18 = self.BatchTrain(parm18,self.max_index,'patch conhesion')
        # CutRateAll,CutNumAll,DiedAll,UnchangeAll,ShrinkAll,SplitAll,SDEArea,SDEFlat,SDETan
        gamma19,x19,y19,X19,Y19 = self.BatchTrain(parm19,self.max_index-1,'The number of patch decay rate')
        # gamma20,x20,y20,X20,Y20 = self.BatchTrain(parm20,self.max_index-1,'CutNumAll')
        # gamma21,x21,y21,X21,Y21 = self.BatchTrain(parm21,self.max_index-1,'DiedAll')
        # gamma22,x22,y22,X22,Y22 = self.BatchTrain(parm22,self.max_index-1,'UnchangeAll')
        # gamma23,x23,y23,X23,Y23 = self.BatchTrain(parm23,self.max_index-1,'ShrinkAll')
        # gamma24,x24,y24,X24,Y24 = self.BatchTrain(parm24,self.max_index-1,'SplitAll')
        # gamma25,x25,y25,X25,Y25 = self.BatchTrain(parm25,self.max_index,'SDEArea')
        gamma26,x26,y26,X26,Y26 = self.BatchTrain(parm26,self.max_index,'SDEFlat')
        gamma27,x27,y27,X27,Y27 = self.BatchTrain(parm27,self.max_index,'SDETan')
        # # ShrinkAll,SplitAll
        # # #plt.subplot(1,5,1)
        # # #plt.scatter(x1,y1)
        # plt.plot(X1,Y1,label='PatchNum')
        # # #plt.subplot(1,5,2)
        # # #plt.scatter(x2,y2)
        # plt.plot(X2,Y2,label='CutRateAll')
        # # #plt.subplot(1,5,3)
        # # #plt.scatter(x3,y3)
        # plt.plot(X3,Y3,label='rasterDNAll')     
        # # #plt.subplot(1,5,4)
        # # #plt.scatter(x4, y4)
        # # # print(X4,Y4)
        # plt.plot(X4,Y4,label='LpiAll')
        # # #plt.subplot(1,5,5)
        # # #plt.scatter(x5, y5)
        # # # print(X5,Y5)
        # plt.plot(X5,Y5,label='Conhesion')
        # plt.plot(X6,Y6,label='ShrinkAll')
        # plt.plot(X7,Y7,label='SplitAll')
        # plt.legend(loc='upper right')
        # pathfig = os.path.join(self.citypath+'//Index.jpg')
        # plt.savefig(pathfig)
        # plt.close()
        # filename = str(self.start) + '_' + str(self.max_index)+'new.csv'
        # with open(filename,'a') as f:
        #     line = str(self.citycode)+ ',' +str(gamma3)+',' +str(gamma12)+',' +str(gamma13)+\
        #     ',' +str(gamma14)+ ',' +str(gamma18)+',' +str(gamma19)+',' +str(gamma26)+\
        #     ',' +str(gamma27)+'\n'
        #     f.write(line)
        #     f.close()
        # with open(filename,'a') as f:
        #     line = str(self.citycode)+ ',' +str(gamma1)+',' +str(gamma2)+',' +str(gamma3)+\
        #     ',' +str(gamma4)+ ',' +str(gamma5)+',' +str(gamma6)+',' +str(gamma7)+\
        #     ',' +str(gamma8)+ ',' +str(gamma9)+',' +str(gamma10)+',' +str(gamma11)+\
        #     ',' +str(gamma12)+ ',' +str(gamma13)+',' +str(gamma14)+',' +str(gamma15)+\
        #     ',' +str(gamma16)+ ',' +str(gamma17)+',' +str(gamma18)+',' +str(gamma19)+\
        #     ',' +str(gamma20)+ ',' +str(gamma21)+',' +str(gamma22)+',' +str(gamma23)+\
        #     ',' +str(gamma24)+ ',' +str(gamma25)+',' +str(gamma26)+',' +str(gamma27)+'\n'
        #     f.write(line)
        #     f.close()
        
        
    def CSVWrite(self,IndexName,IndexData,citycode):
        with open('ALLIndex//'+IndexName+'.csv','a') as f:
            line =str(citycode)+','
            for index in IndexData:
                line = line + str(index) + ','
            f.write(line+'\n')
        # print(IndexName+'is ok')

    def StatusData(self,citycode):
        CA,maxdn,cityDNAll,rasterDNAll = self.CalcDN()
        PatchNum,PatchSumAll,MpaAll,MpsAll,SiAll,CovAll,CiAveAll,CiRaAll,CiSdAll,LpiAll,PdAll,DivisionAll = self.CalcSingle()
        FracAveAll,FracRaAll,FracSdAll,ConhesionAll = self.CalcVectorIndex()
        CutRateAll,CutNumAll,DiedAll,UnchangeAll,ShrinkAll,SplitAll = self.CalcIndex()
        SDEArea,SDEFlat,SDETan = self.CalcSDE()
        # PatchNum	CutRateAll	rasterDNAll	FracSdAll	LpiAll

        self.BatchPlot(cityDNAll,rasterDNAll,PatchNum,PatchSumAll,MpaAll,MpsAll,SiAll,CovAll,\
        CiAveAll,CiRaAll,CiSdAll,LpiAll,PdAll,DivisionAll,FracAveAll,FracRaAll,FracSdAll,ConhesionAll,\
        CutRateAll,CutNumAll,DiedAll,UnchangeAll,ShrinkAll,SplitAll,SDEArea,SDEFlat,SDETan)

        # NDivisionAll = self.SinglePlot(np.divide(MpsAll,np.max(MpsAll)),self.max_index,'DivisionAll')
        # NConhesionAll = self.SinglePlot(np.divide(ConhesionAll,np.max(ConhesionAll)),self.max_index,'ConhesionAll')
        # NcityDNAll = self.SinglePlot(cityDNAll,self.max_index,'cityDNAll')
        # NcityDNAll = self.SinglePlot(PatchNum,self.max_index,'cityDNAll')

        # self.CSVWrite('SDEArea',SDEArea,citycode)
        # self.CSVWrite('SDEFlat',SDEFlat,citycode)
        # self.CSVWrite('SDETan',SDETan,citycode)

if __name__ == "__main__":
    citycode = sys.argv[-1]
    path = r'E:\TeacherZ\NTL_Landscape\CodePython\Data\citytif'
    filename = os.path.join(path,citycode+'.tif') 
    print(filename)
    Light = Light(filename,citycode)

    # Batch Light Relationship
    # Light.BatchGenerate()

    ## calc vector Index
    Light.StatusData(citycode) #总的Index生成与file生成



    


    





