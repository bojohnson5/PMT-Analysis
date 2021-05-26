void max_amplitudes(string file) {
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

  file.resize(file.size() - 5);
  string title = "Run " + file + " max amplitudes;ADC;Counts";
	
	TH1D* h = new TH1D("h", title.c_str(), 100, 0, 1000);
	double max_amp = 0;
	double amp = 0;
	int total_entries = t->GetEntries();
	for (int i = 0; i < total_entries; i++) {
		t->GetEntry(i);
		max_amp = 0;
		for (int j = 0; j < waveform->size(); j++) {
			amp = (waveform->at(j) - baseline) * polarity;
			if (amp > max_amp)
				max_amp = amp;
		}
		h->Fill(max_amp);
	}
	h->Draw();
	
}
