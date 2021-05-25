int pulsing(vector<UShort_t>* waveform, double thre, 
    double baseline, double polarity, int wind_st = 0, int wind_en = 0);

void pre_post_pulsing(string file, double spe_thre, int pre_low,
    int pre_high, int late_low, int late_high, int after_low, int after_high) {

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

  int pre_counts = 0;
  int spe_counts = 0;
  int late_counts = 0;
  int after_counts = 0;
  bool pulse;
  for (int i = 0; i < num_ent; i++) {
    pulse = false;
    t->GetEntry(i);
    for (int j = 0; j < waveform_size; j++) {
      double adc = (waveform->at(j) - baseline) * polarity;
      if (!pulse && adc > spe_thre){
        pulse = true;
        spe_counts++;
        pre_counts += pulsing(waveform, 0.3 * spe_thre, baseline, polarity, 
            j - pre_high, j - pre_low);
        late_counts += pulsing(waveform, 0.3 * spe_thre, baseline, polarity, 
            j + late_low, j + late_high);
        after_counts += pulsing(waveform, 0.3 * spe_thre, baseline, polarity,
            j + after_low, j + after_high);
      }
    }
  }
  
  cout << "Pre-Pulsing %: " << static_cast<double>(pre_counts) / spe_counts * 100 << '\n';
  cout << "Late-Pulsing %: " << static_cast<double>(late_counts) / spe_counts * 100 << '\n';
  cout << "After-Pulsing %: " << static_cast<double>(after_counts) / spe_counts * 100 << '\n';


}

int pulsing(vector<UShort_t>* waveform, double thre, 
    double baseline, double polarity, int wind_st = 0, int wind_en = 0) {
  Int_t waveform_size = waveform->size();

  if (wind_en > waveform_size) {
    wind_en = waveform_size;
  }

  if (wind_st < 0) {
    wind_st = 0;
  }

  // for each waveform count the # of pulses above threshold
  int counts = 0;
  bool pulse = false;
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

  return counts;
}
