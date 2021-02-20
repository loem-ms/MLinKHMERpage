#!/usr/bin/env python
# coding: utf-8

# # Backpropagation (BP)

# ក្នុងអត្ថបទមុនយើងបានសិក្សាអំពីដំណើរការរៀនដើម្បីកំណត់តម្លៃប៉ារ៉ាម៉ែត្រនៃFNNតាមរយៈវិធីសាស្រ្ត stochastic gradient descend(SGD)។ ដូចដែលអ្នកអាចចាប់អារម្មណ៍បានក្នុងSGD ការគណនាgradientឬដេរីវេនៃអនុគមន៍កម្រិតលម្អៀងត្រូវបានធ្វើឡើង។ ជាទូទៅការគណនានេះអាចធ្វើបានតាមរយៈការគណនាតម្លៃប្រហែលនៃដេរីវេដោយប្រើតម្លៃអនុគមន៍ផ្ទាល់។ ប៉ុន្តែការគណនាបែបនេះចំណាយពេលច្រើន។ ដើម្បីគណនាgradientឬដេរីវេនៃអនុគមន៍ប្រកបដោយប្រសិទ្ធភាព វិធីសាស្រ្តគណនាgradientដោយប្រើ backpropagation ត្រូវបានប្រើប្រាស់ជាទូទៅ។

# ## ភាពលំបាកក្នុងការគណនា gradient

# នៅក្នុង stochastic gradient descend ការគណនាតម្លៃ gradient  នៃអនុគមន៍កម្រិត
# លម្អៀង $(\nabla E\left(\pmb{w}\right)=\partial E\left(\pmb{w}\right)/\partial\pmb{w})$គឺជាដំណាក់កាលដ៏សំខាន់។ ចំពោះFNNច្រើនថ្នាក់ ការគណនា gradient ចំពោះប៉ារ៉ាម៉ែត្រមានភាពស្មុគស្មាញខ្លាំង។ 
# ជាឧទាហរណ៍តម្លៃកម្រិតលម្អៀងចំពោះគម្រូទិន្នន័យសម្រាប់រៀន $\pmb{x}_n$ នៃចំណោទតម្រែតម្រង់(regression) កំណត់ដោយ $E_n=\frac{1}{2}||x_n-t_n||^2$។ យើងសាកល្បងគណនាដេរីវេធៀបនឹងប៉ារ៉ាម៉ែត្រទំងន់ផ្ទាល់ $w_{ji}^{\left(l\right)}$ នៃថ្នាក់ទី $l$។ 
# 

# ដំបូងយើងពិនិត្យឃើញថា 
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}=\left(\pmb{y}\left(\pmb{x}_n\right)-\pmb{t}_n\right)^\top\frac{\partial\pmb{y}}{\partial w_{ji}^{\left(l\right)}}
# $$
# 

# បន្ទាប់មកទៀតយើងត្រូវគណនាដេរីវេ 
# 
# $$
# \frac{\partial\pmb{y}}{\partial w_{ji}^{\left(l\right)}}
# $$

#  ដោយលទ្ធផលនៃFNN $\pmb{y}\left(\pmb{x}\right)$ កំណត់ដោយទម្រង់ខាងក្រោម នោះយើងអាចមើលឃើញបានថាការគណនាដេរីវេតាមវិធីបែបនេះមិនមានប្រសិទ្ធភាពឡើយ ពោលគឺត្រូវចំណាយពេលច្រើនក្នុងការគណនាដោយប្រើProgramming។ 

# $$
# \pmb{y}\left(\pmb{x}\right)=\pmb{f}\left(\pmb{u}^{\left(L\right)}\right)
# $$
# $$
# =\pmb{f}\left(\pmb{W}^{\left(L\right)}\pmb{z}^{\left(L-1\right)}+\pmb{b}^{\left(L\right)}\right) 
# $$
# $$
# =\pmb{f}\left(\pmb{W}^{\left(L\right)}\pmb{f}\left(\pmb{W}^{\left(L-1\right)}\pmb{z}^{\left(L-2\right)}+\pmb{b}^{\left(L-1\right)}\right)+\pmb{b}^{\left(L\right)}\right)
# $$
# $$
# =\pmb{f}\left(\pmb{W}^{\left(L\right)}\pmb{f}\left(\pmb{W}^{\left(L\right)}\pmb{f}(\cdots\pmb{f}(\pmb{W}^{\left(l\right)}\pmb{z}^{\left(l-1\right)}+\pmb{b}^{\left(l\right)}\right)\cdots))+\pmb{b}^{\left(L\right)}\right)
# $$
# 

# វិធីសាស្រ្ត backpropagation អាចជួយដោះស្រាយបញ្ហាគណនាដេរីវេនៃអនុគមន៍បណ្តាក់ច្រើនជាន់បែបនេះបាន។ នៅក្នុងការបង្ហាញខាងក្រោម ដើម្បីសម្រួលដល់ការសរសេរ យើងកំណត់សរសេរតួ bias ជាផ្នែកមួយនៃប៉ារ៉ាម៉ែត្រទំងន់ផ្ទាល់នៃណឺរ៉ូនដែរ ពោលគឺ $w_{0j}^{\left(l\right)}=b_j^{\left(l\right)}$។ ហេតុនេះដោយកំណត់ណឺរ៉ូនទី0នៃថ្នាក់$(l-1)$ ឱ្យបញ្ចេញនូវលទ្ធផល $z_0^{\left(l-1\right)}=1$ ជានិច្ចនោះយើងអាចសរសេរលទ្ធផលនៃណឺរ៉ូនដោយទម្រង់ខាងក្រោម។
# 
# $$
# u_j^{\left(l\right)}=\sum_{i=1}^{k}{w_{ji}^{\left(l\right)}z_i^{\left(l-1\right)}}+b_j=\sum_{i=0}^{k}{w_{ji}^{\left(l\right)}z_i^{\left(l-1\right)}}
# $$

# ![BP-1](images/BP-1.png)

# ## Computational Graph and Backpropogation

# ដើម្បីសម្រួលដល់ការស្វែងយល់អំពី Backpropagation Method, យើងនឹងសិក្សាអំពីក្រាបតំណាងប្រមាណវិធីនិងដំណើរការBackpropagationជាមុនសិន។
# 
# ពិនិត្យលើការធ្វើប្រមាណវិធីងាយដូចរូបខាងក្រោម។ រូបនេះបង្ហាញពីការគណនាថ្លៃចំណាយលើការទិញកុំព្យូទ័រតម្លៃ\$1000ចំនួន1 គ្រឿងនិងសៀវភៅតម្លៃ\$50ចំនួន3 ក្បាល រួមទាំងការបង់ពន្ធ10%។ 
# 
# ![graph-cal-1](images/graph-cal-1.png)

# ដោយប្រើក្រាបតាងប្រមាណវិធី យើងអាចទទួលបានអត្ថប្រយោជន៍សំខាន់២យ៉ាង។ ដំបូងគឺការសម្រួលដល់ការស្វែងយល់ដំណាក់កាលគណនាទោះបីជាការធ្វើប្រមាណវិធីក្នុងបញ្ហាស្មុគស្មាញក្តី។ នៅទីនេះដោយការផ្តាច់ប្រមាណវិធីធំជាបង្គុំនៃប្រមាណវិធីងាយៗច្រើនបន្តបន្ទាប់គ្នា យើងអាចសម្រួលដល់ការគណនានិងការស្វែងយល់ពីតម្លៃលេខនៅតាមដំណាក់កាលនីមួយៗបាន។
# 
# អត្ថប្រយោជន៍ទីពីរគឺការសិក្សាលើបម្រែបម្រួល ពោលគឺ ដេរីវេ ។
# 
# ពិចារណាលើសំណួរ ថាតើ ថ្លៃចំណាយសរុបក្នុងការទិញទំនិញខាងលើនឹងប្រែប្រួលយ៉ាងណានៅពេលដែលតម្លៃកុំព្យូទ័រ ឬ តម្លៃសៀវភៅប្រែប្រួល\$1? សំណួរនេះមានន័យស្មើនឹងការគណនាដេរីវេលើថ្លៃចំណាយសរុបធៀបនឹងតម្លៃកុំព្យូទ័រ ឬ ធៀបនឹងតម្លៃសៀវភៅដែរ។ ពោលបើយើង​តាង​$L$ជាថ្លៃចំណាយសរុប និង $x$ជាតម្លៃកុំព្យូទ័រនោះ ការគណនាបម្រែបម្រួលថ្លៃចំណាយសរុបពេលតម្លៃកុំព្យូទ័រប្រែប្រួល១ឯកតាគឺជាតម្លៃនៃ $\frac{\partial L}{\partial x}$។
# 
# យើងនឹងពន្យល់ពីការគណនាជាក្រោយ ប៉ុន្តែដោយប្រើ Computational Graph ដូចរូបខាងក្រោមយើងអាចបានចម្លើយយ៉ាងងាយ។ 
# 
# ![graph-cal-2](images/graph-cal-2.png)

# តាមក្រាបខាងលើនេះ ព្រួញពណ៌ក្រហមតំណាងអោយតម្លៃដេរីវេនៃអនុគមន៍ធាតុចេញចុងក្រោយ(output)ធៀបនឹងអថេរនៅតាមដំណាក់កាលនីមួយៗនៃការគណនា។ ក្នុងករណីខាងលើ គេអាចនិយាយបានថា ថ្លៃចំណាយសរុបនឹងប្រែប្រួល1.10នៅពេលតម្លៃកុំព្យូទ័រប្រែប្រួល១ឯកតា និង 3.30នៅពេលតម្លៃសៀវភៅប្រែប្រួល១ឯកតា។ 
# 
# ដំណើរការគណនា output ដោយប្រើតម្លៃ input និងប្រមាណវិធីជាបន្តបន្ទាប់គ្នាដូចរូបខាងលើ(ព្រួញខ្មៅ)ហៅថា forward propogation រីឯការគណនាបម្រែបម្រួល(ដេរីវេ)នៃ output ធៀបនឹងអថេរនៅតាមដំណាក់កាលនីមួយៗនៃការគណនា ហៅថា backforward។
# 
# 

# ពេលនេះ យើងពិនិត្យលើរបៀបគណនាវិញ។ 
# 
# ជាទូទៅចំពោះការគណនាតម្លៃ $y=f(x)$ យើងអាចបង្ហាញជា computational graph ដូចរូបខាងក្រោម។
# 
# ![graph-cal-3](images/graph-cal-3.png)
# 

# ដើម្បីគណនាដេរីវេនៃអនុគមន៍output ធៀបនឹងអថេរ input យើងត្រូវធ្វើប្រមាណវិធីគុណរវាងតម្លៃដែលបញ្ជូនតាម backward propogation ​ពីផ្នែកoutput និង អនុគមន៍តាងដេរីវេនោះ។ នៅទីនេះតម្លៃដែលបញ្ជូនតាមbackward propogationគឺជាតម្លៃនៃដេរីវេរបស់outputធៀបនឹងដំណាក់កាលខាងក្រោយនៃការគណនា។ 
# 
# វិធីគណនាបែបនេះគឺអនុវត្តតាមវិធាន​ Chain Rule នៃការធ្វើដេរីវេលើអនុគមន៍បណ្តាក់។ 
# 
# ឧទាហរណ៍ បើយើងសិក្សាលើអនុគមន៍ 
# 
# $$
# z = t^2 ,  t= x+y
# $$
# 
# នោះ 
# 
# $$
# \frac{\partial  z}{\partial x} = \frac{\partial  z}{\partial t}\frac{\partial  t}{\partial x}
# $$
# 
# $$
# \frac{\partial  z}{\partial y} = \frac{\partial  z}{\partial t}\frac{\partial  t}{\partial y}
# $$
# 
# បើយើងបង្ហាញវាជា Compuational Graph យើងនឹងបានទម្រង់ដូចខាងក្រោម។ 

# ![graph-cal-4](images/graph-cal-4.png)

# នៅទីនេះ តម្លៃព្រួញក្រហមបង្ហាញពីដំណាក់កាលនីមួយៗនៃការធ្វើដេរីវេរបស់outputធៀបអថេរនានាក្នុងប្រមាណវិធី។ ដូចដែលអ្នកអាចកត់សម្គាល់បាន ដោយប្រើ Chain Rule និង Computational Graph យើងអាចគណនាដេរីវេនៃប្រមាណវិធីស្មុគស្មាញបានយ៉ាងងាយតាមលំនាំគម្រូបញ្ជូនបន្ត backward ពីផ្នែកoutput ទៅផ្នែក inputវិញ។ នេះហើយជាមូលហេតុនៃការហៅថា backpropogation។ 
# 
# ភាពងាយស្រួលនៃវិធីនេះគឺត្រង់ថា យើងអាចអោយComputerគណនាដេរីវេដោយស្វ័យប្រវត្តិបាន(automatic differential, AD) ព្រោះដំណើរការគណនាគ្រាន់តែបញ្ជូនបន្តនូវចម្លៃនៃដេរីវេតាមថ្នាក់នីមួយៗច្រាស់ទិសដៅនៃinput។ ម្យ៉ាងដោយប្រើ Compuation Graph, ការគណនាដេរីវេនៅតាមដំណាក់កាលនីមួយៗអាចធ្វើបានយ៉ាងងាយដូចគម្រូខាងក្រោម។ 
# 
# ![graph-cal-add](images/graph-cal-5.png)
# 
# ![graph-cal-mul](images/graph-cal-6.png)
# 

# ## ការគណនាតាម backpropagation ករណីFNNមានប៉ារ៉ាម៉ែត្រពីរថ្នាក់(ណឺរ៉ូន៣ថ្នាក់)
# 
# ![BP-1](images/BP-1.png)

# រូបខាងលើបង្ហាញទម្រង់នៃFNNមានប៉ារ៉ាម៉ែត្រពីរថ្នាក់(ណឺរ៉ូន៣ថ្នាក់)នៃចំណោទតម្រែតម្រង់
# (regression)។ អនុគមន៍សកម្ម(activation function) នៃថ្នាក់លទ្ធផលចុងក្រោយកំណត់ដោយអនុគមន៍identity $\left(f\left(x\right)=x\right)$។ 
# 

# សន្មតធាតុចូលនៃបណ្តាញនេះដោយ $\pmb{x}=\left[x_1\ x_2\ x_3\ x_4\right]^\top$។ លទ្ធផលនៃថ្នាក់ណឺរ៉ូនទីមួយពោលថ្នាក់ធាតុចូលគឺ $z_i^{\left(1\right)}=x_i$ និងលទ្ធផលនៃថ្នាក់កណ្តាល$z_j^{\left(2\right)}$ ព្រមទាំងលទ្ធផលនៃថ្នាក់លទ្ធផលចុងក្រោយ$y_j\left(\pmb{x}\right)=\ z_j^{\left(3\right)}$ កំណត់ដោយទម្រង់ខាងក្រោម។

# $$
# z_j^{\left(2\right)}=f\left(u_j^{\left(2\right)}\right)=f\left(\sum_{i}{w_{ji}^{(2)}z_i^{\left(1\right)}}\right)
# $$
# 
# $$
# y_j\left(\pmb{x}\right)=z_j^{\left(3\right)}=u_j^{\left(3\right)}=\sum_{i}{w_{ji}^{(3)}z_i^{\left(2\right)}}
# $$

# សន្មតយកផលបូកការេនៃលម្អៀងជាអនុគមន៍កម្រិតលម្អៀងនៃបណ្តាញនេះ។ 
# 
# $$
# E_n=\frac{1}{2}||y_j(\pmb{x}_n)-t_n||^2=\frac{1}{2}\sum_j {\left(y_j(\pmb{x}_n)-t_n\right)^2}
# $$

# ពេលនេះយើងពិនិត្យលើការគណនាដេរីវេនៃអនុគមន៍នេះធៀបនឹងប៉ារ៉ាម៉ែត្ររបស់វា។

# ដំបូងយើងគណនាដេរីវេធៀបប៉ារ៉ាម៉ែត្រនៃផ្នែកថ្នាក់លទ្ធផលចុងក្រោយនៃបណ្តាញ 
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(3\right)}}
# $$

# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(3\right)}}=\frac{\partial}{\partial w_{ji}^{\left(3\right)}}\left\{\frac{1}{2}||y(\pmb{x}_n)-t_n||^2\right\}=(y(\pmb{x}_n)-t_n)^\top \frac{\partial y}{\partial w_{ji}^{\left(3\right)}}
# $$
# 

# ដោយ 
# 
# $$
# y_j\left(\pmb{x}\right)=z_j^{\left(3\right)}=u_j^{\left(3\right)}=\sum_{i}{w_{ji}^{(3)}z_i^{\left(2\right)}}
# $$

# នោះ
# 
# $$
# \frac{\partial\pmb{y}}{\partial w_{ji}^{\left(3\right)}}=\left[0\cdots0\ z_i^{\left(2\right)}\ 0\cdots0\right]^\top
# $$

# ដូច្នេះ 
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(3\right)}}=\left(\pmb{y}\left(\pmb{x}_n\right)-\pmb{t}_n\right)^\top\frac{\partial\pmb{y}}{\partial w_{ji}^{\left(3\right)}}=\left(y_j\left(\pmb{x}_n\right)-t_{nj}\right)z_i^{\left(2\right)}
# $$
# 

# បន្ទាប់ពីនេះ យើងពិនិត្យលើថ្នាក់កណ្តាល $\frac{\partial E_n}{\partial w_{ji}^{\left(2\right)}}$
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(2\right)}}=\frac{\partial E_n}{\partial u_j^{\left(2\right)}}\frac{\partial u_j^{\left(2\right)}}{\partial w_{ji}^{\left(2\right)}}
# $$

# ដោយ 
# 
# $$
# u_j^{\left(2\right)}=\sum_{i}{w_{ji}^{\left(2\right)}z_i^{\left(1\right)}}
# $$
# 
# នោះយើងបាន
# 
# $$
# \frac{\partial u_j^{\left(2\right)}}{\partial w_{ji}^{\left(2\right)}}=z_i^{\left(1\right)}
# $$
# 

# ម្យ៉ាងទៀត បើយក $k$ ជាចំនួនណឺរ៉ូននៅ ថ្នាក់លទ្ធផលចុងក្រោយនោះ $E_n$ មាន $u_1^{\left(3\right)},\cdots,u_k^{\left(3\right)}$ ជាអថេរដែលយើងអាចសរសេរអនុគមន៍ដេរីវេដូចខាងក្រោម។

# $$
# \frac{\partial E_n}{\partial u_j^{\left(2\right)}}=\sum_{k}\frac{\partial E_n}{\partial u_k^{\left(3\right)}}
# \frac{\partial u_k^{\left(3\right)}}{\partial u_j^{\left(2\right)}}
# $$
# 
# $$
# \frac{\partial E_n}{\partial u_k^{\left(3\right)}}=\frac{\partial}{\partial u_k^{\left(3\right)}}\left\{\frac{1}{2}\sum_{j}\left(y_j\left(\pmb{x}_n\right)-\pmb{t}_n\right)^2\right\}
# $$
# 
# $$
# =\frac{\partial}{\partial u_k^{\left(3\right)}}\left\{\frac{1}{2}\sum_{j}\left(u_j^{\left(3\right)}-\pmb{t}_n\right)^2\right\}=u_k^{\left(3\right)}-t_{nk}
# $$
# 

# ដោយ 
# 
# $$
# \frac{\partial u_k^{\left(3\right)}}{\partial u_j^{\left(2\right)}}=\frac{\partial}{\partial u_j^{\left(2\right)}}\left\{\sum_{i}{w_{ki}^{\left(3\right)}z_i^{\left(2\right)}}\right\}
# $$
# 
# $$
# =\frac{\partial}{\partial u_j^{\left(2\right)}}\left\{\sum_{i}{w_{ki}^{\left(3\right)}f\left(u_j^{\left(2\right)}\right)}\right\}
# $$
# 
# $$
# =w_{kj}^{\left(3\right)}f^\prime\left(u_j^{\left(2\right)}\right)
# $$
# 

# ហេតុនេះ យើងបាន 
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(2\right)}}=\left(f^\prime\left(u_j^{\left(2\right)}\right)\sum_{k}{w_{kj}^{\left(3\right)}\left(u_k^{\left(3\right)}-t_{nk}\right)}\right)z_i^{\left(1\right)}
# $$

# ## ករណីទូទៅ៖ FNNច្រើនថ្នាក់

# ![BP-2](images/BP-2.png)

# ដំបូង យើងពិនិត្យលើប៉ារ៉ាម៉ែត្រ $w_{ji}^{\left(l\right)}$ នៃថ្នាក់ទី $l$ ។ 
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}=\frac{\partial E_n}{\partial u_j^{\left(l\right)}}\frac{\partial u_j^{\left(l\right)}}{\partial w_{ji}^{\left(l\right)}}
# $$
# 

# ដោយពិនិត្យរូបឆ្វេង បម្រែបម្រួលនៃ$u_j^{\left(l\right)}$ ជះឥទ្ធិពលលើតម្លៃនៃ$E_n$ តាមរយៈតម្លៃនៃ$z_j^{\left(l\right)}$ និងតម្លៃលទ្ធផលនៃណឺរ៉ូននៅថ្នាក់ទី$\left(l+1\right)$។ ហេតុនេះយើងបាន
# 
# $$
# \frac{\partial E_n}{\partial u_j^{\left(l\right)}}=\sum_{k}{\frac{\partial E_n}{\partial u_k^{\left(l+1\right)}}\frac{\partial u_k^{\left(l+1\right)}}{\partial u_j^{\left(l\right)}}}
# $$
# 

# ដោយពិនិត្យលើកន្សោមខាងលើ យើងឃើញថា $\partial E_n/\partial u_j^{\left(\bullet\right)}$ បង្ហាញនៅអង្គទាំងសង្ខាង។ នៅទីនេះយើងសន្មតតាង
# 
# $$
# \delta_j^{\left(l\right)}\equiv\frac{\partial E_n}{\partial u_j^{\left(l\right)}}
# $$
# 

# ដោយប្រើទំនាក់ទំនង 
# 
# $$
# u_k^{\left(l+1\right)}=\sum_{j}{w_{kj}^{\left(l+1\right)}z_j^{\left(l\right)}}=\sum_{j}{w_{kj}^{\left(l+1\right)}f\left(u_j^{\left(l\right)}\right)}
# $$ 
# 
# យើងបាន 
# 
# $$
# \partial u_k^{\left(l+1\right)}/\partial u_j^{\left(l\right)}=
# w_{kj}^{\left(l+1\right)}f^\prime\left(u_j^{\left(l\right)}\right)
# $$  
# 
# ។ ហេតុនេះ 
# 
# $$
# \delta_j^{\left(l\right)}=\frac{\partial E_n}{\partial u_j^{\left(l\right)}}=\sum_{k}{\frac{\partial E_n}{\partial u_k^{\left(l+1\right)}}\frac{\partial u_k^{\left(l+1\right)}}{\partial u_j^{\left(l\right)}}}=\sum_{k}{\delta_k^{\left(l+1\right)}\left(w_{kj}^{\left(l+1\right)}f\left(u_j^{\left(l\right)}\right)\right)}
# $$
# 

# តាមទំនាក់ទំនងនេះបង្ហាញថាយើងអាចគណនា $\delta_j^{\left(l\right)}$បានដោយប្រើតម្លៃនៃ$\delta_k^{\left(l+1\right)}\ \left(k=1,2,\ldots\right)$។មានន័យថា បើយើងដឹងតម្លៃ$\delta$ របស់ថ្នាក់ខាងលើពោលគឺ$(l+1)$  នោះយើងអាចគណនាតម្លៃ $\delta$ នៅថ្នាក់ក្រោមបន្តបន្ទាប់តាមទំនាក់ទំនងនេះ។ រូបស្តាំ បង្ហាញពីដំណើរការនៃការគណនា$\delta$ ពីថ្នាក់លើឆ្ពោះទៅថ្នាក់ក្រោមបែបនេះ។ ទំនាក់ទំនងខាងលើនេះពិតជានិច្ចចំពោះគ្រប់ថ្នាក់ទាំងអស់នៃបណ្តាញ។

# ដូច្នេះបើ $\delta$ នៅថ្នាក់លទ្ធផលចុងក្រោយត្រូវបានគណនា នោះ យើងអាចគណនា $\delta$ ពោលគឺដេរីវេនៃប៉ារ៉ាម៉ែត្រនៅថ្នាក់ក្រោមជាបន្តបន្ទាប់បានដោយអនុវត្តតាមទំនាក់ទំនងងាយខាងលើ។ ដោយសារតែលំដាប់នៃការគណនា $\delta$ នៅទីនេះមានទិសដៅផ្ទុយពីទិសដៅបញ្ជូនសញ្ញាណក្នុងការប៉ាន់ស្មានលទ្ធផលរបស់បណ្តាញ ដូចនេះទើបគេហៅឈ្មោះវិធីសាស្រ្តនេះថាជា backpropagation។ 

# ត្រលប់មកផ្នែកនៅសល់នៃ $\frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}=\frac{\partial E_n}{\partial u_j^{\left(l\right)}}\frac{\partial u_j^{\left(l\right)}}{\partial w_{ji}^{\left(l\right)}}$ 
# 
# ដោយ 
# $\frac{\partial E_n}{\partial u_j^{\left(l\right)}}$ អាចគណនាបានដោយគណនា $\delta$ ដូច្នេះ យើងពិនិត្យលើ $\frac{\partial u_j^{\left(l\right)}}{\partial w_{ji}^{\left(l\right)}}$
# 
# តាមទំនាក់ទំនង $u_j^{\left(l\right)}=\sum_{i=0}^{k}{w_{ji}^{\left(l\right)}z_i^{\left(l-1\right)}}$ នោះ $\frac{\partial u_j^{\left(l\right)}}{\partial w_{ji}^{\left(l\right)}}=z_i^{\left(l-1\right)}$ ហេតុនេះយើងបាន  
# 
# $$
# \frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}=\delta_j^{\left(l\right)}z_i^{\left(l-1\right)}
# $$
# 

# ដូចបង្ហាញក្នុងទំនាក់ទំនងដែលទាញបាននេះ ការគណនាដេរីវេដោយផ្នែកនៃអនុគមន៍កម្រិតលម្អៀងធៀបប៉ារ៉ាម៉ែត្រ$w_{ji}^{\left(l\right)}$ដែលភ្ជាប់ណឺរ៉ូនទី $i$ នៃថ្នាក់ទី$\left(l-1\right)$និងណឺរ៉ូនទី $j$ នៃថ្នាក់ទី$(l)$ អាចគណនាបានយ៉ាងងាយដោយប្រើ $\delta_j^{\left(l\right)}$ នៃណឺរ៉ូនទី$j$ នៃថ្នាក់ទី$(l)$ និងលទ្ធផល$z_i^{\left(l-1\right)}$នៃណឺរ៉ូនទី $i$ នៃថ្នាក់ទី$\left(l-1\right)$។ ដូចបានបញ្ជាក់ខាងលើ $\delta_j^{\left(l\right)} $អាចគណនាបានដោយគណនាជាបន្តបន្ទាប់ពីថ្នាក់ខាងលើតាមទំនាក់ទំនងដែលបានទាញពីខាងដើម។ ក្នុងករណីនេះ $\delta_j^{\left(L\right)}$នៃថ្នាក់ខាងលើបំផុត(ថ្នាក់លទ្ធផលចុងក្រោយ)អាចកំណត់បានដោយ 
# 
# $$
# \delta_j^{\left(L\right)}=\frac{\partial E_n}{\partial u_j^{\left(L\right)}}
# $$
# 
# ដែលការគណនាជាក់ស្តែងប្រែប្រួលទៅតាមប្រភេទនៃអនុគមន៍កម្រិតលម្អៀង(ទៅតាមចំណោទ)។
# 

# ដោយបូកសរុបការគណនាខាងលើ នៅពេលដែលគម្រូទិន្នន័យសម្រាប់រៀន$\left(\pmb{x}_n,\pmb{t}_n\right)$ ត្រូវបានផ្តល់ gradientឬដេរីវេនៃអនុគមន៍កម្រិតលម្អៀងអាចគណនាបានតាមលំដាប់លំដោយខាងក្រោម។ ក្នុងករណីនៃការរៀនជាក្រុមតូច(minibatch) ផលបូកនៃgradient បានមកពីការគណនា
# ចំពោះគម្រូទិន្នន័យនិមួយៗត្រូវបានយកជាgradientនៃក្រុមតូចនោះ
# 
# $$
# \frac{\partial E}{\partial w_{ji}^{\left(l\right)}}=\sum_{n}\frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}
# $$
# 
# 

# Input : គម្រូទិន្នន័យសម្រាប់រៀន $\left(\pmb{x}_n,\pmb{t}_n\right)$
# 
# Output : gradientឬដេរីវេនៃអនុគមន៍កម្រិតលម្អៀង $\frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}\ \ \left(l=2,\ldots,L\right)$
# 
# 1. ដោយយក $\pmb{z}^{\left(1\right)}=\pmb{x}$  គណនាតម្លៃនៃ $\pmb{u}^{\left(l\right)},\pmb{z}^{\left(l\right)}\ \ \left(l=2,\ldots,L\right)$តាមលំដាប់
# 2. គណនា $\delta_j^{\left(L\right)}$(តាមធម្មតាវាត្រូវបានកំណត់ដោយ$\delta_j^{\left(L\right)}=z_j-t_j$)
# 3. ចំពោះថ្នាក់កណ្តាល$\left(l=L-1,\ L-2,\cdots,2\right)$ គណនាតម្លៃ$\delta_j^{\left(l\right)}$
# តាមលំដាប់ដោយ$\delta_j^{\left(l\right)}=\sum_{k}{\delta_k^{\left(l+1\right)}\left(w_{kj}^{\left(l+1\right)}f\left(u_j^{\left(l\right)}\right)\right)}$
# 
# 4. ចំពោះថ្នាក់ទី $l\ \left(l=2,\cdots,L\right)$ គណនាតម្លៃ$\frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}$ ដោយ$\frac{\partial E_n}{\partial w_{ji}^{\left(l\right)}}=\delta_j^{\left(l\right)}z_i^{\left(l-1\right)}$
# 

# ## ការគណនា$\delta_j^{\left(L\right)}$នៃថ្នាក់លទ្ធផលចុងក្រោយ

# ដូចបានបង្ហាញខាងលើ ការគណនា $\delta_j^{\left(L\right)}$ នៃថ្នាក់លទ្ធផលចុងក្រោយអាស្រ័យនឹងប្រភេទអនុគមន៍កម្រិតលម្អៀងដែលប្រើ ពោលគឺអាស្រ័យនឹងប្រភេទនៃចំណោទ។

# ### ករណីចំណោទតម្រែតម្រង់ Regression

# ក្នុងករណីចំណោទតម្រែតម្រង់អនុគមន៍កម្រិតលម្អៀងគឺជាផលបូកការេនៃលម្អៀងលើគម្រូ
# ទិន្នន័យនិមួយៗ។
# 
# $$
# E_n=\frac{1}{2}||y(x_n)-t_n||^2=\frac{1}{2}\sum_j(y_j-t_j)^2
# $$

# ក្នុងករណីនេះ អនុគមន៍សកម្ម(activation)នៃថ្នាក់លទ្ធផលចុងក្រោយនៃFNNគឺជាអនុគមន៍identity: 
# 
# $$
# y_j=z_j^{\left(L\right)}=u_j^{\left(L\right)}
# $$
# 
# ។ ហេតុនេះ $\delta_j^{\left(L\right)}$ នៃថ្នាក់លទ្ធផលចុងក្រោយគឺ
# 
# $$
# \delta_j^{\left(L\right)}=u_j^{\left(L\right)}-t_{nj}=z_j^{\left(L\right)}-t_j=y_j-t_j
# $$
# 
# ពោលគឺគម្លាតរវាងតម្លៃលទ្ធផលនៃបណ្តាញ(ណឺរ៉ូន)និងលទ្ធផលក្នុងទិន្នន័យសម្រាប់រៀន។
# 

# ### ករណីចំណោទចំណាត់ថ្នាក់ក្រុម Classificaiton

# ក្នុងចំណោទចំណាត់ថ្នាក់២ក្រុម អនុគមន៍កម្រិតលម្អៀងត្រូវបានកំណត់ដោយ 
# 
# $$
# E_n=-t\log{y}-\left(1-t\right)\log{\left(1-y\right)}
# $$
# 

# ក្នុងករណីនេះ អនុគមន៍សកម្មនៃថ្នាក់លទ្ធផលចុងក្រោយនៃFNNគឺជាអនុគមន៍ sigmoid ហេតុនេះ
# 
# $$
# y=\frac{1}{1+\exp{\left(-u\right)}}
# $$
# 
# $$
# \frac{dy}{du}=-\frac{\exp{\left(-u\right)}}{\left(1+\exp{\left(-u\right)}\right)^2}=y\left(1-y\right)
# $$
# 

# ដូច្នេះយើងបាន
# 
# $$
# \delta_j^{\left(L\right)}=-\frac{t}{y}\times\frac{dy}{du}-\frac{1-t}{1-y}\left(-\frac{dy}{du}\right)
# $$
# 
# $$
# =-t\left(1-y\right)-\left(1-t\right)y=y-t
# $$
# 

# ក្នុងចំណោទចំណាត់ថ្នាក់ច្រើនក្រុម អនុគមន៍កម្រិតលម្អៀងត្រូវបានកំណត់ដោយ 
# 
# $$
# E_n=-\sum_{k}{t_k\log{y_k}}=-\sum_{k}{t_k\log{\left(\frac{\exp{\left(u_k^{\left(L\right)}\right)}}{\sum_{i}\exp{\left(u_i^{\left(L\right)}\right)}}\right)}}
# $$
# 

# ក្នុងករណីនេះ អនុគមន៍សកម្មនៃថ្នាក់លទ្ធផលចុងក្រោយនៃFNNគឺជាអនុគមន៍ softmax ហេតុនេះ
# 
# $$
# y=\frac{\exp{\left(u_k^{\left(L\right)}\right)}}{\sum_{i}\exp{\left(u_i^{\left(L\right)}\right)}}
# $$

# ដូច្នេះយើងបាន
# 
# $$
# \delta_j^{\left(L\right)}=-\sum_{k}{t_k\frac{1}{y_k}\times\frac{\partial y_k}{\partial u_j^{\left(L\right)}}}
# $$
# 
# $$
# =-t_j\left(1-y_j\right)-\sum_{k\neq j}{t_k\left(-y_j\right)}
# $$
# 
# $$
# =\left(y_j-t_j\right)\sum_{k} t_k
# $$
# 
# $$
# =y_j-t_j\ \left(\because\sum_{k} t_k=1\ (one-hot\ vector)\right)
# $$ 
# 

# តាមលទ្ធផលខាងលើ ទាំងករណីចំណោទតម្រែតម្រង់ ទាំងករណីចំណោទចំណាត់ថ្នាក់
# ក្រុម $\delta_j^{\left(L\right)}$ នៃថ្នាក់លទ្ធផលចុងក្រោយគឺ
# $\delta_j^{\left(L\right)}=y_j-t_j$
# ពោលគឺគម្លាតរវាងតម្លៃលទ្ធផលនៃបណ្តាញ(ណឺរ៉ូន)និងលទ្ធផលក្នុងទិន្នន័យសម្រាប់រៀន។
# 

# In[ ]:




