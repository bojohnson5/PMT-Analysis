void fit_spectrum(string file, Double_t low, Double_t high) {
	TFile* f = new TFile(file.c_str());
	TH1D* hist = (TH1D*)f->Get("hist");

	TF1* fit = new TF1("fit", "gaus", low, high);
	// hist->Fit(fit, "QRN");
  hist->Fit(fit, "R");

	hist->Draw();

  // Get the fitted function and its parameters
  fit = hist->GetFunction("fit");
  Double_t p0, p1, p2;
  p0 = fit->GetParameter(0);
  p1 = fit->GetParameter(1);
  p2 = fit->GetParameter(2);

  cout << "Parameter 0: " << p0 << endl;
  cout << "Parameter 1: " << p1 << endl;
  cout << "Parameter 2: " << p2 << endl;

	file.resize(file.size() - 5);
	string file_name = file + "_p_to_v.txt";
	ofstream ofs(file_name);

	for (int i = 0; i < hist->GetNbinsX(); i++) {
		ofs << hist->GetAt(i) << endl;
	}
	ofs.close();
}
