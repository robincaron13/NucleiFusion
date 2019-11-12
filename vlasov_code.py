# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy import integrate
import time

debut = time.time()

###################### Variables globales ################################


N=4000 #nombre de particule test
tini=0 #fm/c temps initial
tfin=30 # temps final
Npas=20 #nombre de pas temporel
A=100 #A du noyaux seul

A1=200 #A du noyaux 1 pour le cas de collision
A2=200 #A du noyaux 2
distance=8  # distance entre les noyaux
R1=distance
R2=-distance


b=8 # Paramètre d'impact pour une collision de deux noyaux
p_ini=600 #Mev impulsion initiale pour un des deux noyaux

#hbar=1.054*10e-34  # m2*kg*s-1
hbar=197.32 #Mev*fm
#c=3*10**8 # m2*s-1
c=1 
#m_nuc=1.67*10**-27 # kg
m_nuc=938 # Mev


#Ef=10**(-3) # Mev
pf=hbar*((3./2.)*(np.pi**2)*0.17)**(1./3.)#Mev

Rfermi=1 # fm

M=m_nuc*A # Masse
m_part=M/N

PotentialUFactor= 20 # facteur multipliant le potentiel U pour une meiller stabilité des noyaux (un peu arbitraire...)

taille_grille=4 #taille de la grille en nombre de diametre de noyaux
valeur_resolution=1./25. #fixe la résolution

R=(1.12)*(A**(1./3.))    #rayon associé au principe gautte d'eau liquide pour les noyaux prop. à A puissance 1/3

resolution=2*R*valeur_resolution   # resolution de la grille
Ngrille=int(((2*R)/resolution)*taille_grille)  # nombre de "case" dans une direction de grille

#construction de la grille
#R rayon de l'atome
#Ngrille nombre de cadrillage de la grille
#Z_cut est la valeur autour de laquelle on trace la densite2d
#resolution : en unité de taille de noyau
#taille_grille est la taille totale de la grille en nombre d'atome




###################### définition des fonctions ################################




def norm(r):
    return np.sqrt((r[0]**2)+(r[1]**2)+(r[2]**2))


def init(N,A,plotnorm=False,plot3d=False,collision=False):
    global pf

#init : retourne les couples de positions et impulsions des particules tests    
#N : nombre de particules test
#A : nombre de nucléons dans le noyau

    rinit=[]
        
    if collision:
        while len(rinit)<N:
        
            np.random.seed()    
            xinit=np.random.uniform(0,2*R1)
            np.random.seed()
            yinit=np.random.uniform(0,2*R1)
            np.random.seed()
            zinit=np.random.uniform(0,2*R1)
            if np.sqrt((xinit**2)+(yinit**2)+(zinit**2))<=R:
                rinit=rinit+[[xinit,yinit,zinit]]

    else:    
        while len(rinit)<N:
        
            np.random.seed()    
            xinit=np.random.uniform(-R,R)
            np.random.seed()
            yinit=np.random.uniform(-R,R)
            np.random.seed()
            zinit=np.random.uniform(-R,R)
            if np.sqrt((xinit**2)+(yinit**2)+(zinit**2))<=R:
                rinit=rinit+[[xinit,yinit,zinit]]

    pinit=[]

    for i in range(len(rinit)):
                        
        px=np.random.uniform(-pf,pf)
        py=np.random.uniform(-pf,pf)
        pz=np.random.uniform(-pf,pf)
            
        pinit=pinit+[[px,py,pz]]            

    if plotnorm:
        for i in range(N):
            plt.plot(norm(rinit[i]),norm(pinit[i]),'b o')
            plt.xlabel('r')
            plt.ylabel('p')
            plt.show()
        
    if plot3d:
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        rx=[]
        for i in range(len(rinit)):
            rx=np.append(rx,rinit[i][0])

        ry=[]
        for i in range(len(rinit)):
            ry=np.append(ry,rinit[i][1])

        rz=[]
        for i in range(len(rinit)):
            rz=np.append(rz,rinit[i][2])
                
        ax.scatter(rx, ry, rz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
                
        ax.set_xlim([-6*10**-15,6*10**-15])
        ax.set_ylim([-6*10**-15,6*10**-15])
        ax.set_zlim([-6*10**-15,6*10**-15])
        plt.show()
            
    return rinit,pinit


def init2(N,A,R1,impact_parameter,impulsion_ini,plotnorm=False,plot3d=False,collision=False):

    #init2 : retourne les couples de positions et impulsions des particules tests avec un paramètre d'impact et une impulsion initiale non nulle    
    #N : nombre de particules test
    #A : nombre de nucléons dans le noyau

    
    global pf
    R=(1.12)*(A**(1./3.))

    rinit=[]
        
    if collision:
        while len(rinit)<N:
        
            np.random.seed()    
            xinit=R1 + np.random.uniform(-R,R)
            np.random.seed()
            yinit=R1 + np.random.uniform(-R,R)
            np.random.seed()
            zinit=impact_parameter + np.random.uniform(-R,R)

            if np.sqrt(((xinit-R1)**2)+((yinit-R1)**2)+((zinit-impact_parameter)**2))<=R:
                rinit=rinit+[[xinit,yinit,zinit]]

    else:    
        while len(rinit)<N:
        
            np.random.seed()    
            xinit=np.random.uniform(-R,R)
            np.random.seed()
            yinit=np.random.uniform(-R,R)
            np.random.seed()
            zinit=np.random.uniform(-R,R)
            if np.sqrt((xinit**2)+(yinit**2)+(zinit**2)) <= R:
                rinit=rinit+[[xinit,yinit,zinit]]

    pinit=[]
    
    if collision:
        for i in range(len(rinit)):

            px=impulsion_ini + np.random.uniform(-pf,pf)
            py= np.random.uniform(-pf,pf)
            pz= np.random.uniform(-pf,pf)
            
            pinit=pinit+[[px,py,pz]]
    else:
        for i in range(len(rinit)):
                        
            px=np.random.uniform(-pf,pf)
            py=np.random.uniform(-pf,pf)
            pz=np.random.uniform(-pf,pf)
            
            pinit=pinit+[[px,py,pz]]            


    if plotnorm:
        for i in range(N):
            plt.plot(norm(rinit[i]),norm(pinit[i]),'b o')
            plt.xlabel('r')
            plt.ylabel('p')
            plt.show()
        
    if plot3d:
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        rx=[]
        for i in range(len(rinit)):
            rx=np.append(rx,rinit[i][0])

        ry=[]
        for i in range(len(rinit)):
            ry=np.append(ry,rinit[i][1])

        rz=[]
        for i in range(len(rinit)):
            rz=np.append(rz,rinit[i][2])
                
        ax.scatter(rx, ry, rz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
                
        ax.set_xlim([-6*10**-15,6*10**-15])
        ax.set_ylim([-6*10**-15,6*10**-15])
        ax.set_zlim([-6*10**-15,6*10**-15])
        plt.show()
            
    return rinit,pinit


def initialisation_noyaux(N,A1,A2,R1,R2,plot3d=False):

# Initialisation des tirages pour les positions et impulsions des noyaux 1 et 2

    Noyau1=init2(N,A1,R1,impact_parameter=b/2,impulsion_ini=-p_ini,collision=True)
    Noyau2=init2(N,A2,R2,impact_parameter=-b/2,impulsion_ini=p_ini,collision=True)

    if plot3d:
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        rx1=[]
        for i in range(len(Noyau1[0])):
            rx1=np.append(rx1,Noyau1[0][i][0])

        ry1=[]
        for i in range(len(Noyau1[0])):
            ry1=np.append(ry1,Noyau1[0][i][1])

        rz1=[]
        for i in range(len(Noyau1[0])):
            rz1=np.append(rz1,Noyau1[0][i][2])
                

        rx2=[]
        for i in range(len(Noyau2[0])):
            rx2=np.append(rx2,Noyau2[0][i][0])

        ry2=[]
        for i in range(len(Noyau2[0])):
            ry2=np.append(ry2,Noyau2[0][i][1])

        rz2=[]
        for i in range(len(Noyau2[0])):
            rz2=np.append(rz2,Noyau2[0][i][2])

            
        ax.scatter(rx1, ry1, rz1,marker='.')
        ax.scatter(rx2, ry2, rz2,c='red',marker='.')


        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
                

        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        ax.set_zlim([-10,10])

        plt.show()


    return Noyau1,Noyau2



def densite3d(rinit,A,plot_name='',plot2d=False,test_sum=False,projection=True,collision=False):
    #densite3d : retourne et plot la densitée avec les positions des particules test en entrée
    
    matrix_density=np.zeros([Ngrille,Ngrille,Ngrille])    

    for i in range(len(rinit)):
        for m in range(Ngrille):
            #selection de la ligne et la colonne et la superligne dans la grille pour chaque particule test         
            if rinit[i][2]>-taille_grille*R+m*resolution and rinit[i][2]<=-taille_grille*R+(m+1)*resolution:
                    superligne=m
                    for k in range(Ngrille):
                        if rinit[i][0]>-taille_grille*R+k*resolution and rinit[i][0]<=-taille_grille*R+(k+1)*resolution:
                            ligne=k
                            for n in range(Ngrille):
                                if rinit[i][1]>-taille_grille*R+n*resolution and rinit[i][1]<=-taille_grille*R+(n+1)*resolution:
                                    colonne=n
                                    matrix_density[ligne][colonne][superligne]+=1

 
    if test_sum:
        print(np.sum(matrix_density))

    #Pour obtenir la densité on divise le nombre de particule test par case par le nombre de particule
    #total multiplié au volume d'un case
    
    matrix_density=matrix_density/(len(rinit)*(resolution**3))

    if plot2d:
        if projection:
            sum_matrix=np.zeros([Ngrille,Ngrille])  
            for i in range(Ngrille):
                sum_matrix+=matrix_density[:,:,i]
            
            #plt.imshow(sum_matrix,cmap="hot", vmin=0, vmax=0.17)#vmax = densitée critique
            plt.imshow(sum_matrix,cmap="hot", vmin=0, vmax=0.05)#vmax = densitée critique


            liste_xlabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            xlabels = ['{}'.format(liste_xlabel[0]), '{}'.format(liste_xlabel[1]), '{}'.format(liste_xlabel[2]), '{}'.format(liste_xlabel[3]),'{}'.format(liste_xlabel[4]),'{}'.format(liste_xlabel[5])]
            plt.xticks(np.arange(0,Ngrille, Ngrille/6),xlabels)
            
            liste_ylabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            ylabels = ['{}'.format(liste_ylabel[0]), '{}'.format(liste_ylabel[1]), '{}'.format(liste_ylabel[2]), '{}'.format(liste_ylabel[3]),'{}'.format(liste_ylabel[4]),'{}'.format(liste_ylabel[5])]
            plt.yticks(np.arange(0,Ngrille, Ngrille/6),ylabels)
            
        else:
            
            plt.imshow(matrix_density[int(len(matrix_density)/2)],cmap="hot_r",vmin=0, vmax=0.05)
            plt.close("all")

        plt.colorbar()
        plt.xlabel("x (fm)")
        plt.ylabel("z (fm)")

       
        filename='parton-modelisation/'+plot_name+'.png'
        plt.savefig(filename)
    
        #plt.show()
        plt.close("all")


    return matrix_density
        






def densite3d_noyaux(rinit1,rinit2,plot_name='',plot2d=False,test_sum=False,projection=True,collision=False):
    
    #fonction semblable à densite3d mais pour deux noyaux (nécessite donc deux sets de coordonnées en entrée)
    
    matrix_density1=np.zeros([Ngrille,Ngrille,Ngrille])    
    matrix_density2=np.zeros([Ngrille,Ngrille,Ngrille])    

    for i in range(len(rinit1)):
        for m in range(Ngrille):
            
            #selection de la ligne et la colonne et la superligne dans la grille pour chaque particule test         
          
            if rinit1[i][2]>-taille_grille*R+m*resolution and rinit1[i][2]<=-taille_grille*R+(m+1)*resolution:
                superligne=m
                for k in range(Ngrille):
                    if rinit1[i][0]>-taille_grille*R+k*resolution and rinit1[i][0]<=-taille_grille*R+(k+1)*resolution:
                        ligne=k
                        for n in range(Ngrille):
                            if rinit1[i][1]>-taille_grille*R+n*resolution and rinit1[i][1]<=-taille_grille*R+(n+1)*resolution:
                                colonne=n
                                matrix_density1[ligne][colonne][superligne]+=1

    if collision:
        for i in range(len(rinit2)):
            for m in range(Ngrille):
                
                #selection de la ligne et la colonne et la superligne dans la grille pour chaque particule test         
              
                if rinit2[i][2]>-taille_grille*R+m*resolution and rinit2[i][2]<=-taille_grille*R+(m+1)*resolution:
                    superligne=m
                    for k in range(Ngrille):
                        if rinit2[i][0]>-taille_grille*R+k*resolution and rinit2[i][0]<=-taille_grille*R+(k+1)*resolution:
                            ligne=k
                            for n in range(Ngrille):
                                if rinit2[i][1]>-taille_grille*R+n*resolution and rinit2[i][1]<=-taille_grille*R+(n+1)*resolution:
                                    colonne=n
                                    matrix_density2[ligne][colonne][superligne]+=1


    if test_sum:
        print(np.sum(matrix_density1))
        print(np.sum(matrix_density2))

    #Pour obtenir la densité on divise le nombre de particule test par case par le nombre de particule
    #total multiplié au volume d'un case

    matrix_density1=matrix_density1/(len(rinit1)*(resolution**3))
    matrix_density2=matrix_density2/(len(rinit2)*(resolution**3))

    if plot2d:
        if projection:
            sum_matrix=np.zeros([Ngrille,Ngrille])  

            for i in range(Ngrille):
                sum_matrix+=matrix_density1[:,i,:]+matrix_density2[:,i,:]
              
            plt.imshow(sum_matrix,cmap="hot",vmin=0, vmax=0.05)
            
            liste_xlabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            xlabels = ['{}'.format(liste_xlabel[0]), '{}'.format(liste_xlabel[1]), '{}'.format(liste_xlabel[2]), '{}'.format(liste_xlabel[3]),'{}'.format(liste_xlabel[4]),'{}'.format(liste_xlabel[5])]
            plt.xticks(np.arange(0,Ngrille, Ngrille/6),xlabels)
            
            liste_ylabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            ylabels = ['{}'.format(liste_ylabel[0]), '{}'.format(liste_ylabel[1]), '{}'.format(liste_ylabel[2]), '{}'.format(liste_ylabel[3]),'{}'.format(liste_ylabel[4]),'{}'.format(liste_ylabel[5])]
            plt.yticks(np.arange(0,Ngrille, Ngrille/6),ylabels)
            
        else:
            
            #plt.imshow(matrix_density1[int(len(matrix_density1)/2)],cmap="hot_r",vmin=0, vmax=0.17)
            plt.imshow(matrix_density1[int(len(matrix_density1)/2)],cmap="hot_r",vmin=0, vmax=0.05)

            plt.close("all")

        plt.colorbar()
        plt.xlabel("x (fm)")
        plt.ylabel("z (fm)")
        
        filename='parton-modelisation/'+plot_name+'.png'
        plt.savefig(filename)
    
        #plt.show()
        plt.close("all")


    return matrix_density1+matrix_density2




def gradU(RHO,rhos=0.17,plot_name='',dx=False,dy=False,dz=False,plot2d=False):

     #fonction qui retourne et plot une matrice 3d contenant la dérivée partielle du potentiel lorsque l'on entre en entrée une densité
     #dx , dy , dz : choisir un seul "True" pour définir la direction de la dérivée
    
     RHO_norm=RHO/rhos
     
     #équation du potentiel dirrectement appliqué à la matrice densité :
     
     ################################
     matrix_potential=( -356.*(RHO_norm) + 303.*np.power(RHO_norm,(7./6.)) )*PotentialUFactor
     ################################
     #facteur PotentialUFactor ajouté "à la main" à l'équation pour obtenir une meilleur stabilité des noyaux
     
     if dx:
         #contruit 2 matrice de taille plus grande pour y insérer la matrice potentiel afin de sommer des positions séparer d'une case de la grille
         
         matrix_up=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])
         matrix_down=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])

         matrix_up[:Ngrille,1:Ngrille+1,1:Ngrille+1]=matrix_potential
         matrix_down[2:Ngrille+3,1:Ngrille+1,1:Ngrille+1]=matrix_potential                        
        
         gradU = -(matrix_up-matrix_down)[1:Ngrille+1,1:Ngrille+1,1:Ngrille+1]


     if dy:
         
         matrix_up=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])
         matrix_down=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])

         matrix_up[1:Ngrille+1,:Ngrille,1:Ngrille+1]=matrix_potential
         matrix_down[1:Ngrille+1,2:Ngrille+3,1:Ngrille+1]=matrix_potential                        
         
         gradU = -(matrix_up-matrix_down)[1:Ngrille+1,1:Ngrille+1,1:Ngrille+1]

     if dz:
         
         matrix_up=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])
         matrix_down=np.zeros([Ngrille+2,Ngrille+2,Ngrille+2])
         
         matrix_up[1:Ngrille+1,1:Ngrille+1,:Ngrille]=matrix_potential
         matrix_down[1:Ngrille+1,1:Ngrille+1,2:Ngrille+3]=matrix_potential                        
           
         gradU = -(matrix_up-matrix_down)[1:Ngrille+1,1:Ngrille+1,1:Ngrille+1]
       
    
     if plot2d:
            sum_gradU=np.zeros([Ngrille,Ngrille])  
            for i in range(Ngrille):
                sum_gradU+=gradU[:,:,i]
            
            plt.imshow(sum_gradU,cmap="hot",vmin=0 , vmax=0.05)

            liste_xlabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            xlabels = ['{}'.format(liste_xlabel[0]), '{}'.format(liste_xlabel[1]), '{}'.format(liste_xlabel[2]), '{}'.format(liste_xlabel[3]),'{}'.format(liste_xlabel[4]),'{}'.format(liste_xlabel[5])]
            plt.xticks(np.arange(0,Ngrille, Ngrille/6),xlabels)
            
            liste_ylabel=[str(round((-(Ngrille/2)*resolution),3)),str(round((-(Ngrille/2)*resolution*2/3),3)),str(round(((-(Ngrille/2)*resolution)*1/3),3)),0,str( round((((Ngrille/2)*resolution)*1/3),3)),str(round((((Ngrille/2)*resolution)*2/3),3))]
            ylabels = ['{}'.format(liste_ylabel[0]), '{}'.format(liste_ylabel[1]), '{}'.format(liste_ylabel[2]), '{}'.format(liste_ylabel[3]),'{}'.format(liste_ylabel[4]),'{}'.format(liste_ylabel[5])]
            plt.yticks(np.arange(0,Ngrille, Ngrille/6),ylabels)
        
            plt.colorbar()
            plt.xlabel("x (fm)")
            plt.ylabel("z (fm)")

            filename='parton-modelisation/'+plot_name+'.png'
            plt.savefig(filename)

            #plt.show()
            plt.close("all")

     return gradU



def indice_position(pos):
    
    #fonction qui renvoie les position (ligne, colonne, superligne) lorsque que l'on entre les positions physique en entrée
    
    return abs( int(((-taille_grille*R)-pos)/resolution ))

    
    

def res_temps_verlet(N,Npas,A,tini,tfin,densite=False):
    
    #resolution de l'équation de Vlasov via la méthode de "Verlet" pour un seul noyaux
    #N : nombre de particule test
    #Npas nombre de pas temporel (pas de taille deltat défini plus bas)
    
    
    deltat=(tfin-tini)/Npas
    
    rinit=init(N,A)

    for k in range(0,Npas):


            
        RHO=densite3d(rinit[0],A,plot2d=densite,plot_name='temps t0+{}xdeltat'.format(k),projection=True)
        
        gradU_x=gradU(RHO,dx=True,dy=False,dz=False)
        gradU_y=gradU(RHO,dx=False,dy=True,dz=False)
        gradU_z=gradU(RHO,dx=False,dy=False,dz=True)
        
        
        
       
        for j in range(N):
            pos_x=indice_position(rinit[0][j][0])
            pos_y=indice_position(rinit[0][j][1])
            pos_z=indice_position(rinit[0][j][2])
            
            pos=[pos_x,pos_y,pos_z]
            valideur=0
            for i in range(len(pos)):
                if 0<pos[i] and pos[i]<Ngrille :
                    valideur+=1
                if valideur==3:
                    rxdeltat=rinit[0][j][0]+rinit[1][j][0]/m_nuc*deltat+(1/2)*(-gradU_x[pos_x][pos_y][pos_z])*(deltat**2)
                    rydeltat=rinit[0][j][1]+rinit[1][j][1]/m_nuc*deltat+(1/2)*(-gradU_y[pos_x][pos_y][pos_z])*(deltat**2)
                    rzdeltat=rinit[0][j][2]+rinit[1][j][2]/m_nuc*deltat+(1/2)*(-gradU_z[pos_x][pos_y][pos_z])*(deltat**2)
 
                    rinit[0][j]=[rxdeltat,rydeltat,rzdeltat]

        RHO=densite3d(rinit[0],A)
        
        gradU_x_deltat=gradU(RHO,dx=True,dy=False,dz=False)
        gradU_y_deltat=gradU(RHO,dx=False,dy=True,dz=False)
        gradU_z_deltat=gradU(RHO,dx=False,dy=False,dz=True)

        for n in range(N):
            pos_x=indice_position(rinit[0][n][0])
            pos_y=indice_position(rinit[0][n][1])
            pos_z=indice_position(rinit[0][n][2])
            
            pos=[pos_x,pos_y,pos_z]
            valideur=0
            for i in range(len(pos)):
                if 0<pos[i] and pos[i]<Ngrille :
                    valideur+=1
                if valideur==3:
                    pxdeltat=rinit[1][n][0]+(1/2)*(-gradU_x[pos_x][pos_y][pos_z])*deltat+(1/2)*(-gradU_x_deltat[pos_x][pos_y][pos_z])*deltat
                    pydeltat=rinit[1][n][1]+(1/2)*(-gradU_y[pos_x][pos_y][pos_z])*deltat+(1/2)*(-gradU_y_deltat[pos_x][pos_y][pos_z])*deltat
                    pzdeltat=rinit[1][n][2]+(1/2)*(-gradU_z[pos_x][pos_y][pos_z])*deltat+(1/2)*(-gradU_z_deltat[pos_x][pos_y][pos_z])*deltat
            
                    rinit[1][k]=[pxdeltat,pydeltat,pzdeltat]            
                else:
                    pxdeltat=0
                    pydeltat=0
                    pzdeltat=0
                
                    rinit[1][k]=[pxdeltat,pydeltat,pzdeltat]

        print("pas effectués : ", k+1 , "/{}".format(Npas))

        

def res_temps_verlet_deux_noyaux(N,Npas,A1,A2,R1,R2,tini,tfin,densite=False):
    
    #resolution de l'équation de Vlasov via la méthode de "Verlet" pour un deux noyaux
    #N : nombre de particule test
    #Npas nombre de pas temporel (pas de taille deltat défini plus bas)
    
    
    deltat=(tfin-tini)/Npas   
    
    rinit1,rinit2=initialisation_noyaux(N,A1,A2,R1,R2,plot3d=False)
    
    
    for k in range(0,Npas):

        RHO=densite3d_noyaux(rinit1[0],rinit2[0],plot_name='temps t0+{}xdeltat'.format(k),plot2d=True,projection=True,collision=True)

        gradU_x=gradU(RHO,dx=True,dy=False,dz=False)
        gradU_y=gradU(RHO,dx=False,dy=True,dz=False)
        gradU_z=gradU(RHO,dx=False,dy=False,dz=True)
        
        print(np.max(gradU_x))
       
        for j in range(N):
            x1=indice_position(rinit1[0][j][0])
            y1=indice_position(rinit1[0][j][1])
            z1=indice_position(rinit1[0][j][2])
            
            pos1=[x1,y1,z1]
            valideur1=0

            for i in range(len(pos1)):
                if 0<pos1[i] and pos1[i]<Ngrille :
                    valideur1+=1
                if valideur1==3:      
                    rxdeltat1=rinit1[0][j][0]+rinit1[1][j][0]/m_nuc*deltat+(1/2)*(-gradU_x[x1][y1][z1])*(deltat**2)/m_nuc
                    rydeltat1=rinit1[0][j][1]+rinit1[1][j][1]/m_nuc*deltat+(1/2)*(-gradU_y[x1][y1][z1])*(deltat**2)/m_nuc
                    rzdeltat1=rinit1[0][j][2]+rinit1[1][j][2]/m_nuc*deltat+(1/2)*(-gradU_z[x1][y1][z1])*(deltat**2)/m_nuc
                    
                    rinit1[0][j]=[rxdeltat1,rydeltat1,rzdeltat1]

                    
        for j in range(N):
            x2=indice_position(rinit2[0][j][0])
            y2=indice_position(rinit2[0][j][1])
            z2=indice_position(rinit2[0][j][2])
            
            pos2=[x2,y2,z2]
            valideur2=0

            for i in range(len(pos2)):
                if 0<pos2[i] and pos2[i]<Ngrille :
                    valideur2+=1
                if valideur2==3: 
                    rxdeltat2=rinit2[0][j][0]+rinit2[1][j][0]/m_nuc*deltat+(1/2)*(-gradU_x[x2][y2][z2])*(deltat**2)/m_nuc
                    rydeltat2=rinit2[0][j][1]+rinit2[1][j][1]/m_nuc*deltat+(1/2)*(-gradU_y[x2][y2][z2])*(deltat**2)/m_nuc
                    rzdeltat2=rinit2[0][j][2]+rinit2[1][j][2]/m_nuc*deltat+(1/2)*(-gradU_z[x2][y2][z2])*(deltat**2)/m_nuc
                    rinit2[0][j]=[rxdeltat2,rydeltat2,rzdeltat2]



        RHO=densite3d_noyaux(rinit1[0],rinit2[0],A)
        
        gradU_x_deltat=gradU(RHO,dx=True,dy=False,dz=False)
        gradU_y_deltat=gradU(RHO,dx=False,dy=True,dz=False)
        gradU_z_deltat=gradU(RHO,dx=False,dy=False,dz=True)
        


        for n in range(N):
            x1=indice_position(rinit1[0][n][0])
            y1=indice_position(rinit1[0][n][1])
            z1=indice_position(rinit1[0][n][2])
            
            pos1=[x1,y1,z1]
            valideur1=0
            for i in range(len(pos1)):
                if 0<pos1[i] and pos1[i]<Ngrille :
                    valideur1+=1
                if valideur1==3:
                    pxdeltat1=rinit1[1][n][0]+(1/2)*(-gradU_x[x1][y1][z1])*deltat+(1/2)*(-gradU_x_deltat[x1][y1][z1])*deltat
                    pydeltat1=rinit1[1][n][1]+(1/2)*(-gradU_y[x1][y1][z1])*deltat+(1/2)*(-gradU_y_deltat[x1][y1][z1])*deltat
                    pzdeltat1=rinit1[1][n][2]+(1/2)*(-gradU_z[x1][y1][z1])*deltat+(1/2)*(-gradU_z_deltat[x1][y1][z1])*deltat
     
                    rinit1[1][k]=[pxdeltat1,pydeltat1,pzdeltat1]            

                else:
                    pxdeltat1=0
                    pydeltat1=0
                    pzdeltat1=0
                
                    rinit1[1][k]=[pxdeltat1,pydeltat1,pzdeltat1]
                    
        for n in range(N):
            x2=indice_position(rinit2[0][n][0])
            y2=indice_position(rinit2[0][n][1])
            z2=indice_position(rinit2[0][n][2])
            
            pos2=[x2,y2,z2]
            valideur2=0
            for i in range(len(pos2)):
                if 0<pos2[i] and pos2[i]<Ngrille :
                    valideur2+=1
                if valideur2==3:
                    pxdeltat2=rinit2[1][n][0]+(1/2)*(-gradU_x[x2][y2][z2])*deltat+(1/2)*(-gradU_x_deltat[x2][y2][z2])*deltat
                    pydeltat2=rinit2[1][n][1]+(1/2)*(-gradU_y[x2][y2][z2])*deltat+(1/2)*(-gradU_y_deltat[x2][y2][z2])*deltat
                    pzdeltat2=rinit2[1][n][2]+(1/2)*(-gradU_z[x2][y2][z2])*deltat+(1/2)*(-gradU_z_deltat[x2][y2][z2])*deltat
            
                    rinit2[1][k]=[pxdeltat2,pydeltat2,pzdeltat2]            
                else:
                    pxdeltat2=0
                    pydeltat2=0
                    pzdeltat2=0
                
                    rinit2[1][k]=[pxdeltat2,pydeltat2,pzdeltat2]
                    
                    
                    
                    
        print ("pas effectués : ", k+1 , "/{}".format(Npas))



###################### Main program ################################




"""
anciennes commandes


r0=np.max(init(N,A)[0])
x=np.linspace(-r0,r0,Npas)
xtest=np.linspace(0,1,Npas)
y=np.linspace(-r0,r0,Npas)
X,Y = np.meshgrid(x,y)

r=[r0,r0/2,r0/2]


#d_exp=np.exp(-(X-r0/2)**2-(Y-r0/2)**2)

#plt.colorbar()
#plt.show()

#plt.figure(1)
#plt.imshow(d)print(np.shape(RHO))

#plt.colorbar()
#plt.show()

#plt.contourf(x,y,d,100)
#plt.plot(r[0],r[1],'o')

#plt.figure(300)
#RHO=densite3d(init(N,A)[0],A)[62][62][:]
#plt.plot(potentielU(RHO))

#plt.plot(xtest,potentielU(xtest))


POSITIONS=init(N,A)[0]

xposition=POSITIONS[2][0]
yposition=POSITIONS[2][1]
zposition=POSITIONS[2][2]

RHO=densite3d(POSITIONS,A,plot2d=False)

print(np.shape(gradU(RHO,dx=True)))
plt.imshow(gradU(RHO,dx=True)[41])
"""



res_temps_verlet_deux_noyaux(N,Npas,A1,A2,R1,R2,tini,tfin,densite=True)

#Noyaux=initialisation_noyaux(N,A1,A2,R1,R2,plot3d=True)
#densite3d_noyaux(Noyaux[0][0],Noyaux[1][0],plot2d=True,collision=True)

#plt.colorbar()
#plt.show()



fin = time.time()
#print ("Duree totale: ",fin-debut," s")
