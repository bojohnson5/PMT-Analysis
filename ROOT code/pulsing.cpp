void pulsing(string file, double thre, int wind_st = 0, int wind_en = 0) {
  TFile* f = new TFile(file.c_str());
  TTree* t = (TTree*)f->Get("waveformTree");

	Double_t baseline, polarity, integral;
	vector<UShort_t>* waveform = 0;
	t->SetBranchAddress("waveform", &waveform);
	t->SetBranchAddress("baseline", &baseline);
	t->SetBranchAddress("polarity", &polarity);

  Int_t num_ent = t->GetEntries();
  t->GetEntry(0);
  Int_t waveform_size = waveform->size();
  double up_time = 5.0 * 60.0; // run time in sec

  // for each waveform count the # of pulses above threshold
  int counts = 0;
  bool pulse = false;
  for (int i = 0; i < num_ent; i++) {
    t->GetEntry(i);
    if (wind_st == wind_en) {
      for (int j = 0; j < waveform_size; j++) {
        double adc = (waveform->at(j) - baseline) * polarity;
        if (!pulse && adc > thre) {
          pulse = true;
          counts++;
        }
        
        if (pulse && adc < thre) {
          pulse = false;
        }
      }
    }
    else {
      for (int j = wind_st; j < wind_en; j++) {
        double adc = (waveform->at(j) - baseline) * polarity;
        if (!pulse && adc > thre) {
          pulse = true;
          counts++;
        }
        
        if (pulse && adc < thre) {
          pulse = false;
        }
      }
    }
  }

  cout << "For a threshold of " << thre << " saw "
    << counts << " pulses which is a rate of " << counts / up_time << " Hz\n";
}
