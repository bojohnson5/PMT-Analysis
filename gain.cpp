void gain(string file, int st, int en) {
	TFile* f = new TFile(file.c_str());
	TTree* t = (TTree*)f->Get("waveformTree");
	
	if (t == nullptr) {
		cout << "No tree" << endl;
		
		return;
	}
	
	vector<UShort_t>* waveform = 0;
	Double_t baseline;
	Double_t polarity;
	t->SetBranchAddress("waveform", &waveform);
	t->SetBranchAddress("baseline", &baseline);
	t->SetBranchAddress("polarity", &polarity);

  int num_chann = 4096;
  double peak_v = 2.0;
  double volts[num_chann];
  double dv = peak_v / num_chann;
  volts[0] = 0;
  for (int i = 1; i < num_chann; i++) {
    volts[i] = volts[i - 1] + dv;
  }

  double e = 1.6e-19;

  double integral;
  double volt;
  double dt = 4.0 * 1e-9;
  double resis = 50.0;
  double average_integral = 0;
  for (int i = 0; i < t->GetEntries(); i++) {
    t->GetEntry(i);
    integral = 0;
    for (int j = st; j < en; j++) {
      int adc = (waveform->at(j) - baseline) * polarity;
      if (adc < 0) {
        adc = 0;
      } 
      volt = volts[adc];
      integral += volt / resis * dt; // integral is in coulombs
    }
    // running average for all waveforms
    average_integral = average_integral + (integral - average_integral) / (i + 1);
  }
  cout << volts[0] << endl;
  cout << volts[num_chann - 1] << endl;

  cout << "Average integrated charge: " << average_integral << " C\n";
  cout << "Gain: " << average_integral / e << '\n';
}
