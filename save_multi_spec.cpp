void view_spectrum(string file, Int_t num, Int_t st = 0,
              Int_t en = 0, Int_t num_bins = 150);

void save_multi_spec(string file, Int_t low_st, Int_t low_en,
                     Int_t high_st, Int_t high_en) {
  Int_t index = 0;
  for (Int_t i = low_st; i <= low_en; i += 5) {
    for (Int_t j = high_st; j <= high_en; j += 5) {
      view_spectrum(file, index, i, j);
      index++;
    }
  }
}

void view_spectrum(string file, Int_t num, Int_t st = 0,
                   Int_t en = 0, Int_t num_bins = 150) {
	TFile* f = new TFile(file.c_str());
	TTree* t = (TTree*)f->Get("waveformTree");

	Double_t baseline, polarity, integral, accum;
	Int_t accum_st, accum_en;
	vector<UShort_t>* waveform = 0;
	Bool_t use_accum;
	if (st == 0 && en == 0 ) {
		use_accum = true;
	}
	else {
		use_accum = false;
	}

	t->SetBranchAddress("waveform", &waveform);
	t->SetBranchAddress("baseline", &baseline);
	t->SetBranchAddress("polarity", &polarity);
	t->SetBranchAddress("accumulator1", &accum);
	t->SetBranchAddress("accumulator1Start", &accum_st);
	t->SetBranchAddress("accumulator1End", &accum_en);

	t->GetEntry(0);

  string ext = ".root";
  file.resize(file.size() - ext.size());
  string title = "Run " + file;
	Int_t window_size = accum_en - accum_st + 1;
	Int_t num_ent = t->GetEntries();
	Double_t up_time = waveform->size() * 4.0 * num_ent;
	up_time /= 1e9;

  TCanvas* c1 = new TCanvas("c1", "ADC Spectrum");
	TH1D* hist = new TH1D("hist", title.c_str(),
				num_bins, -1000, 3000);
  hist->GetXaxis()->SetTitle("Integrated ADC");

	if (use_accum) {
		for (Int_t i = 0; i < num_ent; i++) {
			integral = 0;
			t->GetEntry(i);
			integral = (accum - baseline * window_size) * polarity;
			hist->Fill(integral, 1 / up_time);
		}
	}
	else {
		for (Int_t i = 0; i < num_ent; i++) {
			integral = 0;
			t->GetEntry(i);
			for (Int_t j = st; j < en + 1; j++) {
				integral += (waveform->at(j) - baseline)
					* polarity;
			}
			hist->Fill(integral);
		}
	}
	hist->Draw();

  // Tell what the window size is on the histogram
  stringstream ss;
  ss << st << "-" << en << " window";
  TText* text = new TText(1000, 600, ss.str().c_str());
  text->Draw("same");

  // Save the histogram, appending the file number
  ss.str("");
  ss << file << "_" << num << ".pdf";
  c1->SaveAs(ss.str().c_str());
}
