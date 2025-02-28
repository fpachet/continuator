import("stdfaust.lib");

mid_side(l,r) = out : fi.dcblocker,fi.dcblocker
with {
    mid   = l+r;
    diff  = l-r;
    side  = diff*width;
    out   = mid+side, mid-side;
    width = hslider("stereo_width[midi:ctrl 42]",1,0,1,0.007874);
};

// currently unused via being set to 0 modfactor in 'akjrev'
// resulting sound was too smeared in pitch...
smear_delay(del,modfactor) =
  _ : de.sdelay(MAXDELAY,IT,del_len)
with {
    MAXDELAY =  8192;
    IT       =  1024;
    del_len  =  del + del_mod;
    del_mod  =  no.noise : ba.sAndH(ba.pulse(14000)) : ba.line(14000)
                : _*(del*modfactor);
};

akjrev(cutoff,feedback)
    = (si.bus(2*N) :> si.bus(N) : delaylines(N)) ~
      (delayfilters(N,cutoff,feedback) : feedbackmatrix(N))
with {
    N = 16;
    delays = (2113,2267,2411,2543,2663,2851,2957,3089,3271,3449,3593,3739,3863,4003,4159,4327);
    delayval(i) = ba.take(i+1,delays);
    delaylines(N) = par(i,N,(smear_delay(delayval(i),0.000)));  // 0.0004
    delayfilters(N,cutoff,feedback) = par(i,N,filter(i,cutoff,feedback));
    feedbackmatrix(N) = bhadamard(N);
    bhadamard(2) = si.bus(2) <: +,-;
    bhadamard(n) = si.bus(n) <: (si.bus(n):>si.bus(n/2)) , ((si.bus(n/2),(si.bus(n/2):par(i,n/2,*(-1)))) :> si.bus(n/2))
                   : (bhadamard(n/2) , bhadamard(n/2));
    filter(i,cutoff,feedback) = fi.lowpass(1,cutoff) : *(feedback)
                                :> /(sqrt(N)) : _;
};

akjrev_demo = _,_ : mid_side <: ef.dryWetMixerConstantPower(wet*0.9, akjrev(cutoff,feedback)) :> _,_
with {
    cutoff   = hslider("cutoff[midi:ctrl 45]",7563,500,12000,90.55118);
    feedback = hslider("feedback[midi:ctrl 46]",0.55,0,1,0.00787);
    wet      = hslider("wet[midi:ctrl 47]",0.5,0,1,0.01);
};