// Plot a specified waveform from a ROOT file

void view_waveform(string file, int num_waveform) {
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
	
	t->GetEntry(num_waveform);
	Int_t waveform_size = waveform->size();
	Double_t x[waveform->size()];
	Double_t y[waveform->size()];
	for (int i = 0; i < waveform_size; i++) {
		x[i] = i * 4; // convert to ns
		y[i] = (waveform->at(i) - baseline) * polarity;
	}
	
	TGraph* gr = new TGraph(waveform_size, x, y);
	gr->GetXaxis()->SetTitle("Time [ns]");
	gr->GetYaxis()->SetTitle("ADC");
  file.resize(file.size() - 5);
	stringstream ss;
	ss << "Run " << file << " Waveform " << num_waveform;
	gr->SetTitle(ss.str().c_str());
	gr->Draw();
}
