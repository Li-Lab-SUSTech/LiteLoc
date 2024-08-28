%%
%
% Copyright (c) 2022 Li Lab, Southern University of Science and Technology, Shenzhen.
% 
%%
function [SXY,beadpos,parameters]=calibrate_abr_map(p)

if ~isfield(p,'xrange')
    p.xrange=[-inf inf]; p.yrange=[-inf inf]; 
end


if ~isfield(p,'smoothxy')
    p.smoothxy=0;
end

if ~isfield(p,'isglobalfit')
    p.isglobalfit=0;
end

if ~isfield(p,'filechannel')
    p.filechannel=1;
end

%get bead positions
p.status.String='Load files and segment beads';drawnow

if ~isfield(p,'tabgroup')
    f=figure('Name','Bead stacks calibration');
    p.tabgroup=uitabgroup(f);
    calibrationfigure=f;
else
    calibrationfigure=p.tabgroup.Parent;
end
%get beads from images
[beads,p]=images2beads_globalfit(p);
imageRoi=p.roi{1};

%get positions of beads
for k=length(beads):-1:1
    beadposx(k)=beads(k).pos(1);
    beadposy(k)=beads(k).pos(2);
end

%if only to take beads in a certain range, remove others
if isfield(p,'fov')&&~isempty(p.fov)
    indbad=beadposx<p.fov(1)| beadposx>p.fov(3)|beadposy<p.fov(2)|beadposy>p.fov(4);
    beads=beads(~indbad);
end

if isempty(beads)
    warndlg('Could not find and segment any bead. ROI size too large?')
    p.status.String='error: could not find and segment any bead...';
    return
end

p.midpoint=round(size(beads(1).stack.image,3)/2); %reference for beads
p.ploton=false;

f0g=p.midpoint;
for k=1:length(beads)
    beads(k).f0=f0g;
end

%get positions of beads
for k=length(beads):-1:1
    beadposxs(k)=beads(k).pos(1);
    beadposys(k)=beads(k).pos(2);
    beadfilenumber(k)=beads(k).filenumber;
end

%spatially dependent calibration
tgmain=p.tabgroup;
for X=1:length(p.xrange)-1
    for Y=1:length(p.yrange)-1
        if length(p.xrange)>2||length(p.yrange)>2
            ht=uitab(tgmain,'Title',['X' num2str(X) 'Y' num2str(Y)]);
            p.tabgroup=uitabgroup(ht);
        end
        
        indgood=beadposxs< p.xrange(X+1) & beadposxs>p.xrange(X) & beadposys<p.yrange(Y+1) & beadposys>p.yrange(Y);
        beadsh=beads(indgood);
        
        for k=1:max(beadfilenumber)
            indfile=(beadfilenumber==k)&indgood;
            p.fileax(k).NextPlot='add';
            scatter(p.fileax(k),beadposxs(indfile),beadposys(indfile),60,[1 1 1]);
            scatter(p.fileax(k),beadposxs(indfile),beadposys(indfile),50);
        end
        if isempty(beadsh)
            disp(['no beads found in part' num2str(p.xrange(X:X+1)) ', ' num2str(p.yrange(Y:Y+1))])
            continue
        end
        
        indgoodc=true(size(beadsh));
        gausscal=[];
        p.ax_z=[];

        % get beads calibration
        p.status.String='get beads calibration';drawnow
        [csplinecal,indgoods,beadpos{X,Y},~,testallrois,beadspos]=getstackcal_g(beadsh(indgoodc),p);
        
        for f=1:max(beadpos{X,Y}.filenumber(:))
            indfile=(beadpos{X,Y}.filenumber==f);
            p.fileax(f).NextPlot='add';
            plot(p.fileax(f),beadpos{X,Y}.xim(indfile),beadpos{X,Y}.yim(indfile),'m+');
        end
        
        icf=find(indgoodc);
        icfs=icf(indgoods);
        for k=1:length(csplinecal.cspline.coeff)
            cspline.coeff{k}=single(csplinecal.cspline.coeff{k});
        end
        cspline.dz=csplinecal.cspline.dz;
        cspline.z0=csplinecal.cspline.z0;
        cspline.x0=csplinecal.cspline.x0;
        cspline.global.isglobal=p.isglobalfit;
        cspline.mirror=csplinecal.cspline.mirror;

        gausscal=[];
        gauss_sx2_sy2=[];
        gauss_zfit=[];
        p.ax_sxsy=[];
            
        cspline_all=csplinecal;
        cspline_all=[];
        PSF=csplinecal.PSF;
%         SXY(X,Y)=struct('cspline',cspline,'posind',[X,Y],'EMon',p.emgain,'PSF',{PSF});
        
        SXY(X,Y)=struct('gausscal',gausscal,'cspline_all',cspline_all,'gauss_sx2_sy2',gauss_sx2_sy2,'gauss_zfit',gauss_zfit,...
            'cspline',cspline,'Xrangeall',p.xrange+imageRoi(1),'Yrangeall',p.yrange+imageRoi(2),'Xrange',p.xrange([X X+1])+imageRoi(1),...
            'Yrange',p.yrange([Y Y+1])+imageRoi(2),'posind',[X,Y],'EMon',p.emgain,'PSF',{PSF});        
        
        % ZERNIKE fitting    
        axzernike=axes(uitab(p.tabgroup,'Title','Zernikefit'));
        axPupil=axes(uitab(p.tabgroup,'Title','Pupil'));
        axMode=axes(uitab(p.tabgroup,'Title','ZernikeModel'));   
       
        stack=csplinecal.PSF{1}; %this would be the average... not sure if good.
        mp=ceil(size(stack,1)/2);
        rxy=floor(p.ROIxy/2);
        zborder=round(100/p.dz); %from alignment: outside is bad.
        stack=stack(mp-rxy:mp+rxy,mp-rxy:mp+rxy,zborder+1:end-zborder);

        %arbitrary
        stack=stack*1000; %random photons, before normalized to maximum pixel
%         psfrescale = p.psfrescale;

        p.zernikefit.dz=p.dz;
        
        [SXY(X,Y).zernikefit,PSFZernike,aberrations_avg1,RRSE,MAPE]=zernikefitBeadstack(stack,p.zernikefit,axzernike,axPupil,axMode);
        coeffZ=Spline3D_interp(PSFZernike);
        axzernikef=axes(uitab(p.tabgroup,'Title','Zval'));
        p.z0=size(coeffZ,3)/2;
        posbeads=testfit_spline(testallrois,{coeffZ},0,p,{},axzernikef);

    end
end

axcrlb=axes(uitab(p.tabgroup,'Title','CRLB'));
plotCRLBcsplinePSF(csplinecal.cspline,p.zernikefit,axcrlb);
parameters=myrmfield(p,{'tabgroup','status','ax_z','ax_sxsy','fileax'});

cspline_psf_model.coeff=SXY.cspline.coeff;
cspline_psf_model.x0=SXY.cspline.x0;
cspline_psf_model.z0=SXY.cspline.z0;
cspline_psf_model.dz=SXY.cspline.dz;

vector_psf_model.zernikefit.NA=SXY.zernikefit.NA;
vector_psf_model.zernikefit.refmed=SXY.zernikefit.refmed;
vector_psf_model.zernikefit.refcov=SXY.zernikefit.refcov;
vector_psf_model.zernikefit.refimm=SXY.zernikefit.refimm;
vector_psf_model.zernikefit.wavelength=SXY.zernikefit.lambda;
vector_psf_model.zernikefit.pixelSizeX=SXY.zernikefit.pixelSizeX;
vector_psf_model.zernikefit.pixelSizeY=SXY.zernikefit.pixelSizeY;
vector_psf_model.zernikefit.psfrescale=SXY.zernikefit.psfrescale;
vector_psf_model.zernikefit.Npupil=SXY.zernikefit.Npupil;
vector_psf_model.zernikefit.psfSizeX=SXY.zernikefit.sizeX;
vector_psf_model.zernikefit.psfSizeY=SXY.zernikefit.sizeY;
vector_psf_model.zernikefit.zrang=[-SXY.zernikefit.dz*(SXY.zernikefit.sizeZ+1)/2,SXY.zernikefit.dz*(SXY.zernikefit.sizeZ+1)/2];

vector_psf_model.aberrations=SXY.zernikefit.aberrations;


save(p.outputfile,'SXY','parameters','cspline_psf_model','vector_psf_model','-v7');
filefig=strrep(p.outputfile,'.mat','.fig');
savefig(calibrationfigure,filefig,'compact');
p.status.String='Calibration done';drawnow





