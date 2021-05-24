void find_pv(TH1D* hist);
void find_spe_res_and_peak(TF1* f1, TF1* f2);

void anaylsis(string file, Double_t wind_st, Double_t wind_en,
		Int_t bins = 150, Double_t fit_low1 = -200,
		Double_t fit_high1 = 200, Double_t fit_low2 = 500,
		Double_t fit_high2 = 900) {

	TFile* f = new TFile(file.c_str());
	TTree* t = (TTree*)f->Get("waveformTree");

	Double_t baseline, polarity, integral;
	vector<UShort_t>* waveform = 0;
	t->SetBranchAddress("waveform", &waveform);
	t->SetBranchAddress("baseline", &baseline);
	t->SetBranchAddress("polarity", &polarity);

	t->GetEntry(0);
	Int_t tot_num = t->GetEntries();
	Double_t up_time = waveform->size() * 4.0 * tot_num; // convert to ns
	up_time /= 1e9; // convert to sec

	// histogram title
	string ext = ".root";
	file.resize(file.size() - ext.size());
	string title = "Run " + file + ";Integrated ADC";
	TH1D* hist = new TH1D("hist", title.c_str(), bins, -1000, 3000);

	for (Int_t i = 0; i < tot_num; i++) {
		integral = 0;
		t->GetEntry(i);
		for (Int_t j = wind_st; j < wind_en + 1; j++) {
			integral += (waveform->at(j) - baseline) *
				polarity;
		}
		hist->Fill(integral);
	}

	// fit histogram with 2 gaussians
	TF1* fit1 = new TF1("fit1", "gaus", fit_low1, fit_high1);
	TF1* fit2 = new TF1("fit2", "gaus", fit_low2, fit_high2);
	hist->Fit(fit1, "RQ");
	hist->Fit(fit2, "RQ+");

	hist->Draw();

	find_pv(hist);
	find_spe_res_and_peak(fit1, fit2);

	// set axis back to normal
	hist->GetXaxis()->SetRangeUser(-1000, 3000);
}

void find_pv(TH1D* hist) {
	hist->GetXaxis()->SetRangeUser(50, 1000);
	Double_t max = hist->GetMaximum();
	Int_t max_bin = hist->GetMaximumBin();
	Double_t min = hist->GetMinimum();
	Double_t pv = max / min;
	cout << "P/V:\t\t" << pv << endl;

	// PV for 0.3 SPE, from Hamamatsu spec sheet
	// taking 0.3 SPE to be 0.3 * peak SPE
	Double_t min_03 = hist->GetBinContent(hist->GetXaxis()->FindBin(max * 0.3));
	cout << "P/V (0.3SPE):\t" << max / min_03 << endl;
}

void find_spe_res_and_peak(TF1* f1, TF1* f2) {
	Double_t mean1 = f1->GetParameter(1);
	Double_t mean2 = f2->GetParameter(1);
	Double_t var2 = f2->GetParameter(2);
	Double_t res = (mean2 - mean1) / var2;
	cout << "Resolution:\t" << res << endl;
	cout << "SPE peak:\t" << (mean2 - mean1) << endl;
}
