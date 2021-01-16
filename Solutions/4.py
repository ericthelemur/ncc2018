from CipherUtils import *

print("4A")
text = """
JUR DYKR NLH VB JUR BLZR. JUR IYLDX ZKHRKZ VH NCGYA SLZCKH LH JUR VB-UCKHR ZKHRKZ CS DGVZR LJ BRN HDCJYLBA PLGA. JUR UVHJCGP ICCXH (CG LJ YRLHJ, NVXVERAVL) JCYA ZR JULJ VJ ULA IRRB HRJ KE IP VBHERDJCG BRLZR LBA ED GLBALYY HCZR JVZR LSJRG LEGVY 1875 LH L NLP JC KHR JUR EGVHCBRGH’ EGCERGJP HJCGR JC VBHJGKDJ BRN GRDGKVJH VB JUR LGJ CS ARJRDJVCB. JUR BLZR SVGHJ LEERLGH VB EKIYVD VB LB LGJVDYR VB JUR CIHRGMRG BRNHELERG VB 1877, LBA JUR UVHJCGVLBH HLP VJ NLH DCVBRA LH L TGVHYP URLAYVBR IP L WCKGBLYVHJ NUC NLH HBKIIRA IP BRLZR, IKJ V LZ BCJ HC HKGR. VJ AVAB’J HRRZ LB LDDVARBJ JULJ ACKTYLH IYLDX NLH YCCXVBT JC HRJ KE L JCE HRDGRJ LGDUVMR VB YCBACB LJ JULJ JVZR, LBA HDCJYLBA PLGA TLMR UVZ JUR ERGSRDJ YCDLJVCB. VJ NLH DRBJGLY LBA RLHP JC LDDRHH. VJ ULA HRMRGLY RBJGLBDRH, NVJU L HJRLAP SYCN CS MVHVJCGH, BCJ LYY CS JURZ RBJVGRYP GREKJLIYR, JULJ NCKYA AVHTKVHR JUR DCZVBTH LBA TCVBTH CS JUR LTRBJH LBA CSSVDRGH CS IYLDX’H BRJNCGX. IRHJ CS LYY VJ ULA JUR RBJVGR ZRJGCECYVJLB ECYVDR SCGDR HJLBAVBT TKLGA, NUVDU NLH TGRLJ SCG IYLDX, LBA JCKTU CB ZR. VJ NCKYA IR JUR VARLY YCDLJVCB SCG JUR HULACN LGDUVMR.
HDCJYLBA PLGA ULA ZCMRA JC BRN EGRZVHRH HRMRGLY JVZRH HVBDR VJ NLH SVGHJ HRJ KE, IKJ V NLH IRJJVBT JULJ JUR HULACN LGDUVMR ULA ZCMRA NVJU VJ, LBA V HERBJ JUR IRHJ ELGJ CS JUGRR NRRXH HDCKJVBT JUR BCGZLB HULN IKVYAVBTH LBA JUR BRN “BRN HDCJYLBA PLGA” BROJ ACCG LJ JUR DKGJVH TGRRB IKVYAVBT. VJ VH BCJ RLHP JC AC JULJ NVJUCKJ IRVBT HECJJRA, URBDR JUR BVTUJ JVZR GRDCBBLVHHLBDR VB JUR GLVB. JUR NLJRG LBA JUR DCYA EYLPRA ULMCD NVJU ZP ZCCA, IKJ LYHC NVJU ZLBP ARJRDJCG HPHJRZH HC VJ NLH NCGJU JUR AVHDCZSCGJ.
BC-CBR NLH YVXRYP JC TVMR ZR JUR GRLY EYLBH CS JUR EYLDR, JUCKTU ULGGP NLH ULEEP JC EGCMVAR ZR NVJU L DCEP CS JUR CSSVDVLY YLPCKJ, LBA V KHRA L YVALG HPHJRZ JC ZLE LH ZKDU CS JUR CKJHVAR LH V DCKYA. DCZELGVBT JUR JNC HUCKYA ULMR GRMRLYRA KBZLGXRA HJCGLTR LGRLH LBA TVMRB ZR HCZR VARL UCN V ZVTUJ TRJ VB, IKJ BCJUVBT HUCNRA KE KBJVY V ULA L YKDXP IGRLX. YVJRGLYYP. V ULA YCNRGRA JUR YVALG ARJRDJCG ACNB L HULSJ CB JUR DKGJVH TGRRB GCCS, UCEVBT JC TRJ L TYVZEHR VBJC HCZR CS JUR BRLGIP GCCZH. VJ ULA WKHJ LICKJ GRLDURA TGCKBA YRMRY NURB JUR BPYCB YVBR HBLEERA. VJ ZLAR L URYY CS L GCN LH VJ DYLJJRGRA ACNB, LBA V EGRELGRA JC GKB, IKJ BC LYLGZH NRBJ CSS LBA, ZCGR VZECGJLBJYP, V BCJVDRA HCZRJUVBT HVTBVSVDLBJ. VJ ULA JLXRB LGCKBA SVMR HRDCBAH SCG JUR YVALG JC DGLHU ACNB, NUVDU, LJ L GCKTU RHJVZLJR, ZRLBJ VJ ULA SLYYRB SGRR SCG LGCKBA CBR UKBAGRA LBA JNRBJP ZRJGRH. JULJ NCKYA ULMR JLXRB VJ L YCBT NLP KBARGTGCKBA, LBA JURGR NRGR BC ILHRZRBJ GCCZH ZLGXRA CB JUR EYLBH VB JULJ YCDLJVCB. V NLH EGRJJP HKGR V ULA SCKBA JUR HULACN LGDUVMR: BCN V WKHJ ULA JC SVBA L NLP VB.
"""
Utils.get_stats(text)
key, plain = Substitution.auto_decode(text)
print(Substitution.key_as_str(key), plain)


"""
Substitution Cipher LIDARSTUVWXYZBCEFGHJKMNOPQ (Run-on keyword: LIDAR)
THE CLUE WAS IN THE NAME. THE BLACK MUSEUM IS WORLD FAMOUS AS THE IN-HOUSE MUSEUM OF CRIME AT NEW SCOTLAND YARD. THE HISTORY BOOKS (OR AT LEAST, WIKIPEDIA) TOLD ME THAT IT HAD BEEN SET UP BY INSPECTOR NEAME AND PC RANDALL SOME TIME AFTER APRIL 1875 AS A WAY TO USE THE PRISONERS’ PROPERTY STORE TO INSTRUCT NEW RECRUITS IN THE ART OF DETECTION. THE NAME FIRST APPEARS IN PUBLIC IN AN ARTICLE IN THE OBSERVER NEWSPAPER IN 1877, AND THE HISTORIANS SAY IT WAS COINED AS A GRISLY HEADLINE BY A JOURNALIST WHO WAS SNUBBED BY NEAME, BUT I AM NOT SO SURE. IT DIDN’T SEEM AN ACCIDENT THAT DOUGLAS BLACK WAS LOOKING TO SET UP A TOP SECRET ARCHIVE IN LONDON AT THAT TIME, AND SCOTLAND YARD GAVE HIM THE PERFECT LOCATION. IT WAS CENTRAL AND EASY TO ACCESS. IT HAD SEVERAL ENTRANCES, WITH A STEADY FLOW OF VISITORS, NOT ALL OF THEM ENTIRELY REPUTABLE, THAT WOULD DISGUISE THE COMINGS AND GOINGS OF THE AGENTS AND OFFICERS OF BLACK’S NETWORK. BEST OF ALL IT HAD THE ENTIRE METROPOLITAN POLICE FORCE STANDING GUARD, WHICH WAS GREAT FOR BLACK, AND TOUGH ON ME. IT WOULD BE THE IDEAL LOCATION FOR THE SHADOW ARCHIVE.
SCOTLAND YARD HAD MOVED TO NEW PREMISES SEVERAL TIMES SINCE IT WAS FIRST SET UP, BUT I WAS BETTING THAT THE SHADOW ARCHIVE HAD MOVED WITH IT, AND I SPENT THE BEST PART OF THREE WEEKS SCOUTING THE NORMAN SHAW BUILDINGS AND THE NEW “NEW SCOTLAND YARD” NEXT DOOR AT THE CURTIS GREEN BUILDING. IT IS NOT EASY TO DO THAT WITHOUT BEING SPOTTED, HENCE THE NIGHT TIME RECONNAISSANCE IN THE RAIN. THE WATER AND THE COLD PLAYED HAVOC WITH MY MOOD, BUT ALSO WITH MANY DETECTOR SYSTEMS SO IT WAS WORTH THE DISCOMFORT.
NO-ONE WAS LIKELY TO GIVE ME THE REAL PLANS OF THE PLACE, THOUGH HARRY WAS HAPPY TO PROVIDE ME WITH A COPY OF THE OFFICIAL LAYOUT, AND I USED A LIDAR SYSTEM TO MAP AS MUCH OF THE OUTSIDE AS I COULD. COMPARING THE TWO SHOULD HAVE REVEALED UNMARKED STORAGE AREAS AND GIVEN ME SOME IDEA HOW I MIGHT GET IN, BUT NOTHING SHOWED UP UNTIL I HAD A LUCKY BREAK. LITERALLY. I HAD LOWERED THE LIDAR DETECTOR DOWN A SHAFT ON THE CURTIS GREEN ROOF, HOPING TO GET A GLIMPSE INTO SOME OF THE NEARBY ROOMS. IT HAD JUST ABOUT REACHED GROUND LEVEL WHEN THE NYLON LINE SNAPPED. IT MADE A HELL OF A ROW AS IT CLATTERED DOWN, AND I PREPARED TO RUN, BUT NO ALARMS WENT OFF AND, MORE IMPORTANTLY, I NOTICED SOMETHING SIGNIFICANT. IT HAD TAKEN AROUND FIVE SECONDS FOR THE LIDAR TO CRASH DOWN, WHICH, AT A ROUGH ESTIMATE, MEANT IT HAD FALLEN FREE FOR AROUND ONE HUNDRED AND TWENTY METRES. THAT WOULD HAVE TAKEN IT A LONG WAY UNDERGROUND, AND THERE WERE NO BASEMENT ROOMS MARKED ON THE PLANS IN THAT LOCATION. I WAS PRETTY SURE I HAD FOUND THE SHADOW ARCHIVE: NOW I JUST HAD TO FIND A WAY IN.
"""


print("4B")
text = """
ATRDW PFKFD KITGG TPGTD PRGFG ZGTPP RFGRD PIDZM KYIEQ GTPLR QKYGT PFZHG TPDYD PIKZY GTPZG GZXRY FRDPD PFGWP FFRYL KYGTP YZDGT GTPGT DPPPX BPDZD FRDPR IDZMK YIGTD PRGGZ ZHDKY OWHPY APGTP OZDPK IYZOO KAPFP PXBRD RWQFP LOZAH FFPLP YGKDP WQZYO ZDXKY IRWWK RYAPF RYLHY REWPG ZLPGP DXKYP GTPKD EPFGF GDRGP IQKGT KYVGT PQRDP XKFGR VPYKY GTPKD ZHGWZ ZVGTP QOZAH FGZZX HATZY GDPRG KPFRY LYZGP YZHIT ZYDPR WBZWK GKVRF JZYDZ ATRHM DZGPG TPWRM ZOBZM PDIZJ PDYFG TPMZD WLZOF GRGPF UHFGR FGTPW RMZOI DRJKG QIZJP DYFGT PBTQF KARWM ZDWLR YLMPR DPBPD TRBFG ZZXHA TKYAW KYPLG ZZJPD WZZVG TPDPR WXKIT GZOGT PXZDP BZMPD OHWRY LGTPK YPJKG REKWK GQZOK GFBZW KGKAR WKYOW HPYAP ZHDGR FVYZM KFYZG GZATZ ZFPRO RJZHD PLBRD GYPDG ZAZYG DZWGT PZGTP DFGTR GMRQD KFVFM RDRYR WWKRY APMKW WPYAZ HDRIP RAZHY GPDRW WKRYA PRYLR LRYIP DZHFP FARWR GKZYK YFGPR LZHDF GDRGP IQFTZ HWLEP GZXRK YGRKY RYRLX KGGPL WQHYP RFQBP RAPEP GMPPY GTPPX BPDZD FGZIP GTPDG TPQRD PGTPE KIIPF GBZWK GKARW GTDPR GGZZH DPXBK DPEHG RFPYP XKPFZ OZYPR YZGTP DGTPQ RWFZG TDPRG PYZHD GDRLK YIDZH GPFMP YPPLG ZKYLH APGTP XGZMZ DVGZI PGTPD MTKWP BDPJP YGKYI GTPXO DZXOZ DXKYI RBZMP DEWZA RIRKY FGZHD KYGPD PFGZH DXZFG KXBZD GRYGM PRBZY KYGTK FKFAZ YOHFK ZYRYL XKWLL KFGDH FGGTP EPFGZ HGAZX PMZHW LEPGZ PYAZH DRIPR LQFOH YAGKZ YRWRW WKRYA PEPGM PPYRW WGTDP PGTRG XRVPF KGTRD LOZDG TPPXB PDZDF GZOKI TGZYP RYZGT PDZDH FRYLK YGTPZ GGZXR YADKF KFKGT KYVKF PYFPR YZBBZ DGHYK GQGZL ZUHFG GTRG
""".replace(" ", "")

Utils.get_stats(text)
key, plain = Substitution.auto_decode(text)
print(Substitution.key_as_str(key), plain)

"""
Substitution Cipher: realpoitkuvwxyzbcdfghjmnqs
CHARLES IS RIGHT THE THREATS TO THE EAST ARE GROWING BY THE DAY IN THE SOUTHERN REGION THE OTTOMANS ARE RESTLESS AND IN THE NORTH THE THREE EMPERORS ARE A GROWING THREAT TO OUR INFLUENCE THE FOREIGN OFFICE SEEM PARALYSED FOCUSSED ENTIRELY ON FORMING ALLIANCES AND UNABLE TO DETERMINE THEIR BEST STRATEGY I THINK THEY ARE MISTAKEN IN THEIR OUTLOOK THEY FOCUS TOO MUCH ON TREATIES AND NOT ENOUGH ON REAL POLITIKAS VON ROCHAU WROTE THE LAW OF POWER GOVERNS THE WORLD OF STATES JUST AS THE LAW OF GRAVITY GOVERNS THE PHYSICAL WORLD AND WE ARE PERHAPS TOO MUCH INCLINED TO OVERLOOK THE REAL MIGHT OF THE MORE POWERFUL AND THE INEVITABILITY OF ITS POLITICAL INFLUENCE OUR TASK NOW IS NOT TO CHOOSE A FAVOURED PARTNER TO CONTROL THE OTHERS THAT WAY RISKS WAR AN ALLIANCE WILL ENCOURAGE A COUNTER ALLIANCE AND A DANGEROUS ESCALATION INSTEAD OUR STRATEGY SHOULD BE TO MAINTAIN AN ADMITTEDLY UNEASY PEACE BETWEEN THE EMPERORS TOGETHER THEY ARE THE BIGGEST POLITICAL THREAT TO OUR EMPIRE BUT AS ENEMIES OF ONE ANOTHER THEY ALSO THREATEN OUR TRADING ROUTES WE NEED TO INDUCE THEM TO WORK TOGETHER WHILE PREVENTING THEM FROM FORMING A POWER BLOC AGAINST OUR INTEREST OUR MOST IMPORTANT WEAPON IN THIS IS CONFUSION AND MILD DISTRUST THE BEST OUTCOME WOULD BE TO ENCOURAGE A DYSFUNCTIONAL ALLIANCE BETWEEN ALL THREE THAT MAKES IT HARD FOR THE EMPERORS TO FIGHT ONE ANOTHER OR US AND IN THE OTTOMAN CRISIS I THINK I SENSE AN OPPORTUNITY TO DO JUST THAT
"""
